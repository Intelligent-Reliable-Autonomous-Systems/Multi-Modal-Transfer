"""Implementation of  models for phenological development in WOFOST

Written by: Will Solow, 2025
"""

import datetime

import numpy as np
import torch
from model_engine.inputs.input_providers import DFTensorWeatherDataContainer
from model_engine.inputs.weather_util import daylength
from model_engine.models.base_model import BatchTensorModel
from model_engine.models.states_rates import (
    NDArray,
    ParamTemplate,
    RatesTemplate,
    StatesTemplate,
    Tensor,
    TensorBatchAfgenTrait,
    VariableKiosk,
)
from model_engine.util import EPS


class Vernalisation_TensorBatch(BatchTensorModel):
    """Modification of phenological development due to vernalisation."""

    _force_vernalisation = Tensor(-99.0)  # Bool
    _IS_VERNALIZED = Tensor(-99.0)  # Bool

    class Parameters(ParamTemplate):
        VERNSAT = Tensor(-99.0)
        VERNBASE = Tensor(-99.0)
        VERNRTB = TensorBatchAfgenTrait()
        VERNDVS = Tensor(-99.0)

    class StateVariables(StatesTemplate):
        VERN = Tensor(-99.0)

    class RateVariables(RatesTemplate):
        VERNR = Tensor(-99.0)
        VERNFAC = Tensor(-99.0)

    def __init__(
        self,
        day: datetime.date,
        kiosk: VariableKiosk,
        parvalues: dict,
        device: torch.device,
        num_models: int = 1,
    ) -> None:
        self.num_models = num_models
        super().__init__(day, kiosk, parvalues, device, num_models=self.num_models)

        self.states = self.StateVariables(num_models=self.num_models, kiosk=self.kiosk, VERN=0.0)
        self.rates = self.RateVariables(num_models=self.num_models, kiosk=self.kiosk, publish=["VERNFAC"])

        self._force_vernalisation = torch.zeros((self.num_models,)).to(self.device)
        self._IS_VERNALIZED = torch.zeros((self.num_models,)).to(self.device)

    def calc_rates(self, day: datetime.date, drv: DFTensorWeatherDataContainer, _VEGETATIVE: torch.Tensor) -> None:
        """Compute state rates for integration"""
        r = self.rates
        s = self.states
        p = self.params

        DVS = self.kiosk.DVS
        r.VERNR = torch.where(
            _VEGETATIVE,
            torch.where(
                ~self._IS_VERNALIZED.to(torch.bool),
                torch.where(DVS < p.VERNDVS, p.VERNRTB(drv.TEMP), 0.0),
                0.0,
            ),
            0.0,
        )
        r.VERNFAC = torch.where(
            ~self._IS_VERNALIZED.to(torch.bool),
            torch.where(
                DVS < p.VERNDVS,
                torch.clamp(
                    (s.VERN - p.VERNBASE) / (p.VERNSAT - p.VERNBASE).clamp(min=EPS),
                    torch.tensor([0.0]).to(self.device),
                    torch.tensor([1.0]).to(self.device),
                ),
                1.0,
            ),
            1.0,
        )

        self._force_vernalisation = torch.where(
            _VEGETATIVE,
            torch.where(DVS < p.VERNDVS, self._force_vernalisation, 1.0),
            self._force_vernalisation,
        )
        self.rates._update_kiosk()

    def integrate(self, day: datetime.date, delt: float = 1.0, _VEGETATIVE: torch.Tensor = None) -> None:
        """Integrate state rates"""
        s = self.states
        r = self.rates
        p = self.params

        s.VERN = s.VERN + r.VERNR

        self._IS_VERNALIZED = torch.where(
            _VEGETATIVE,
            torch.where(
                s.VERN >= p.VERNSAT,
                1.0,
                torch.where(self._force_vernalisation.to(torch.bool), 1.0, 0.0),
            ),
            self._IS_VERNALIZED,
        )

        self.states._update_kiosk()

    def reset(self, day: datetime.date, inds: torch.Tensor = None):
        """Reset states and rates"""
        inds = torch.ones((self.num_models,), device=self.device).to(torch.bool) if inds is None else inds

        s = self.states
        r = self.rates

        s.VERN = torch.where(inds, 0.0, s.VERN).detach()

        r.VERNR = torch.where(inds, 0.0, r.VERNR).detach()
        r.VERNFAC = torch.where(inds, 0.0, r.VERNFAC).detach()

        self._force_vernalisation = torch.where(inds, 0.0, self._force_vernalisation).detach()
        self._IS_VERNALIZED = torch.where(inds, 0.0, self._IS_VERNALIZED).detach()

        self.states._update_kiosk()
        self.rates._update_kiosk()

    def get_output(self, va: list[str] = None) -> torch.Tensor:
        """
        Return the output
        """
        if va is None:
            return self.states.VERNDVS
        else:
            output_vars = torch.empty(size=(self.num_models, len(va))).to(self.device)
            for i, v in enumerate(va):
                if v in self.states.trait_names():
                    output_vars[i, :] = getattr(self.states, v)
                elif v in self.rates.trait_names():
                    output_vars[i, :] = getattr(self.rates, v)
            return output_vars

    def get_extra_states(self) -> dict[str, torch.Tensor]:
        """
        Get extra states
        """
        return {
            "_force_vernalisation",
            self._force_vernalisation,
            "_IS_VERNALIZED",
            self._IS_VERNALIZED,
        }

    def set_model_specific_params(self, k: str, v: torch.Tensor) -> None:
        """
        Set the specific parameters to handle overrides as needed
        Like casting to ints
        """
        setattr(self.params, k, v)


class WOFOST_Phenology_TensorBatch(BatchTensorModel):
    """Implements the algorithms for phenologic development in WOFOST."""

    _STAGE_VAL = {
        "sowing": 0,
        "emerging": 1,
        "vegetative": 2,
        "reproductive": 3,
        "mature": 4,
        "dead": 5,
    }
    _STAGE = NDArray(["sowing"])

    class Parameters(ParamTemplate):
        TSUMEM = Tensor(-99.0)
        TBASEM = Tensor(-99.0)
        TEFFMX = Tensor(-99.0)
        TSUM1 = Tensor(-99.0)
        TSUM2 = Tensor(-99.0)
        TSUM3 = Tensor(-99.0)
        IDSL = Tensor(-99.0)
        DLO = Tensor(-99.0)
        DLC = Tensor(-99.0)
        DVSI = Tensor(-99.0)
        DVSM = Tensor(-99.0)
        DVSEND = Tensor(-99.0)
        DTSMTB = TensorBatchAfgenTrait()

        DTBEM = Tensor(-99)

    class RateVariables(RatesTemplate):
        DTSUME = Tensor(-99.0)
        DTSUM = Tensor(-99.0)
        DVR = Tensor(-99.0)
        RDEM = Tensor(-99.0)

    class StateVariables(StatesTemplate):
        DVS = Tensor(-99.0)
        TSUM = Tensor(-99.0)
        TSUME = Tensor(-99.0)
        DATBE = Tensor(-99)

    def __init__(
        self,
        day: datetime.date,
        kiosk: VariableKiosk,
        parvalues: dict,
        device: torch.device,
        num_models: int = 1,
    ) -> None:

        self.num_models = num_models
        self.num_stages = len(self._STAGE_VAL)
        super().__init__(day, kiosk, parvalues, device, num_models=self.num_models)

        DVS = -0.1
        self._STAGE = ["emerging" for _ in range(self.num_models)]

        self.states = self.StateVariables(
            num_models=self.num_models,
            kiosk=self.kiosk,
            publish=["DVS"],
            TSUM=0.0,
            TSUME=0.0,
            DVS=DVS,
            DATBE=0,
        )

        self.rates = self.RateVariables(num_models=self.num_models, kiosk=self.kiosk)

        if torch.any(self.params.IDSL >= 2):
            self.vernalisation = Vernalisation_TensorBatch(day, kiosk, parvalues, device, num_models=self.num_models)

        self.min_tensor = torch.tensor([0.0]).to(self.device)

    def calc_rates(self, day: datetime.date, drv: DFTensorWeatherDataContainer) -> None:
        """Calculates the rates for phenological development"""
        p = self.params
        r = self.rates
        s = self.states

        r.DTSUME = torch.zeros(size=(self.num_models,))
        r.DTSUM = torch.zeros(size=(self.num_models,))
        r.DVR = torch.zeros(size=(self.num_models,))

        DVRED = 1.0
        if torch.any(self.params.IDSL >= 1):
            if hasattr(drv, "DAYL"):
                DAYLP = drv.DAYL
            elif hasattr(drv, "LAT"):
                DAYLP = torch.tensor(daylength(day, drv.LAT)).to(self.device)
            DVRED = torch.clamp(
                self.min_tensor,
                torch.tensor([1.0]).to(self.device),
                (DAYLP - p.DLC) / (p.DLO - p.DLC).clamp(min=EPS),
            )

        VERNFAC = 1.0
        stage_tensor = torch.tensor([self._STAGE_VAL[s] for s in self._STAGE], device=self.device)
        stage_masks = torch.stack([stage_tensor == i for i in range(self.num_stages)])
        (
            self._sowing,
            self._emerging,
            self._vegetative,
            self._reproductive,
            self._mature,
            self._dead,
        ) = stage_masks
        if torch.any(self.params.IDSL >= 2):
            self.vernalisation.calc_rates(day, drv, self._vegetative)
            VERNFAC = self.kiosk.VERNFAC

        r.RDEM = torch.where(self._sowing, torch.where(drv.TEMP > p.TBASEM, 1, 0), 0)

        if hasattr(drv, "TRESP"):
            dtsum_update = torch.clamp(drv.TRESP, self.min_tensor, p.TEFFMX)
        else:
            dtsum_update = torch.where(
                self._emerging, torch.clamp(drv.TEMP - p.TBASEM, self.min_tensor, p.TEFFMX), p.DTSMTB(drv.TEMP)
            )
        r.DTSUME = torch.where(
            self._emerging,
            torch.clamp(self.min_tensor, (p.TEFFMX - p.TBASEM), dtsum_update),
            r.DTSUME,
        )

        r.DTSUM = torch.where(
            self._sowing | self._emerging | self._dead,
            r.DTSUM,
            torch.where(self._vegetative, dtsum_update * VERNFAC * DVRED, dtsum_update),
        )

        r.DVR = torch.where(
            self._sowing | self._dead,
            r.DVR,
            torch.where(
                self._emerging,
                0.1 * r.DTSUME / (p.TSUMEM).clamp(min=EPS),
                torch.where(
                    self._vegetative,
                    r.DTSUM / p.TSUM1.clamp(min=EPS),
                    torch.where(
                        self._reproductive,
                        r.DTSUM / p.TSUM2.clamp(min=EPS),
                        torch.where(self._mature, r.DTSUM / p.TSUM3.clamp(min=EPS), 0),
                    ),
                ),
            ),
        )
        self.rates._update_kiosk()

    def integrate(self, day: datetime.date, delt: float = 1.0) -> None:
        """Updates the state variable and checks for phenologic stages"""

        p = self.params
        r = self.rates
        s = self.states

        if torch.any(self.params.IDSL >= 2):
            self.vernalisation.integrate(day, delt, self._vegetative)

        s.TSUME = s.TSUME + r.DTSUME
        s.DVS = s.DVS + r.DVR
        s.TSUM = s.TSUM + r.DTSUM
        s.DATBE = s.DATBE + r.RDEM

        # Stage transitions
        self._STAGE[(self._sowing & (s.DATBE >= p.DTBEM)).cpu().numpy()] = "emerging"
        self._STAGE[(self._emerging & (s.DVS >= 0.0)).cpu().numpy()] = "vegetative"
        self._STAGE[(self._vegetative & (s.DVS >= 1.0)).cpu().numpy()] = "reproductive"
        self._STAGE[(self._reproductive & (s.DVS >= p.DVSM)).cpu().numpy()] = "mature"
        self._STAGE[(self._mature & (s.DVS >= p.DVSEND)).cpu().numpy()] = "dead"
        self.states._update_kiosk()

    def reset(self, day: datetime.date, inds: torch.Tensor = None) -> None:
        """Reset model"""
        inds = torch.ones((self.num_models,), device=self.device).to(torch.bool) if inds is None else inds

        s = self.states
        r = self.rates
        DVS = -0.1
        self._STAGE = np.where(inds.cpu().numpy(), "emerging", self._STAGE)

        s.DVS = torch.where(inds, DVS, s.DVS).detach()
        s.TSUM = torch.where(inds, 0.0, s.TSUM).detach()
        s.TSUME = torch.where(inds, 0.0, s.TSUME).detach()
        s.DATBE = torch.where(inds, 0.0, s.DATBE).detach()

        r.DTSUME = torch.where(inds, 0.0, r.DTSUME).detach()
        r.DTSUM = torch.where(inds, 0.0, r.DTSUM).detach()
        r.DVR = torch.where(inds, 0.0, r.DVR).detach()
        r.RDEM = torch.where(inds, 0.0, r.RDEM).detach()

        if torch.any(self.params.IDSL >= 2):
            self.vernalisation.reset(day, inds=inds)

        self.states._update_kiosk()
        self.rates._update_kiosk()

    def get_output(self, va: list[str] = None) -> torch.Tensor:
        """
        Return the phenological stage as the floor value
        """
        if va is None:
            return self.states.DVS
        else:
            output_vars = torch.empty(size=(self.num_models, len(va))).to(self.device)
            for i, v in enumerate(va):
                if v in self.states.trait_names():
                    output_vars[i, :] = getattr(self.states, v)
                elif v in self.rates.trait_names():
                    output_vars[i, :] = getattr(self.rates, v)
                elif v in self.kiosk:
                    output_vars[:, i] = getattr(self.kiosk, v)
            return output_vars

    def get_extra_states(self) -> dict[str, torch.Tensor]:
        """
        Get extra states
        """
        return {"_STAGE": self._STAGE}

    def set_model_specific_params(self, k: str, v: torch.Tensor) -> None:
        """
        Set the specific parameters to handle overrides as needed
        Like casting to ints
        """
        setattr(self.params, k, v)
