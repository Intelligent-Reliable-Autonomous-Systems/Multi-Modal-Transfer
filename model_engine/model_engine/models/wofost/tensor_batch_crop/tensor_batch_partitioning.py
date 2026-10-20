"""Implementation of model for partitioning in WOFOST

Written by: Will Solow, 2025
"""

from datetime import date

import torch
from model_engine.inputs.input_providers import DFTensorWeatherDataContainer
from model_engine.models.base_model import BatchTensorModel
from model_engine.models.states_rates import (
    ParamTemplate,
    StatesTemplate,
    Tensor,
    TensorBatchAfgenTrait,
    VariableKiosk,
)
from model_engine.util import EPS


class Partitioning_NPK_TensorBatch(BatchTensorModel):
    """Class for assimilate partitioning based on development stage (`DVS`)
    with influence of NPK stress.
    """

    _THRESHOLD_N_FLAG = Tensor(0.0)
    _THRESHOLD_N = Tensor(0.0)

    class Parameters(ParamTemplate):
        FRTB = TensorBatchAfgenTrait()
        FLTB = TensorBatchAfgenTrait()
        FSTB = TensorBatchAfgenTrait()
        FOTB = TensorBatchAfgenTrait()
        NPART = Tensor(-99.0)
        NTHRESH = Tensor(-99.0)
        PTHRESH = Tensor(-99.0)
        KTHRESH = Tensor(-99.0)

    class StateVariables(StatesTemplate):
        FR = Tensor(-99.0)
        FL = Tensor(-99.0)
        FS = Tensor(-99.0)
        FO = Tensor(-99.0)

    def __init__(
        self,
        day: date,
        kiosk: VariableKiosk,
        parvalues: dict,
        device: torch.device,
        num_models: int = 1,
    ) -> None:
        self.num_models = num_models
        super().__init__(day, kiosk, parvalues, device, num_models=self.num_models)

        k = self.kiosk
        FR = self.params.FRTB(k.DVS)
        FL = self.params.FLTB(k.DVS)
        FS = self.params.FSTB(k.DVS)
        FO = self.params.FOTB(k.DVS)

        self._THRESHOLD_N_FLAG = torch.zeros((self.num_models,)).to(self.device)
        self._THRESHOLD_N = torch.zeros((self.num_models,)).to(self.device)

        self.states = self.StateVariables(
            num_models=self.num_models,
            kiosk=self.kiosk,
            publish=["FR", "FL", "FS", "FO"],
            FR=FR,
            FL=FL,
            FS=FS,
            FO=FO,
        )

    def calc_rates(self, day: date, drv: DFTensorWeatherDataContainer, _emerging: torch.Tensor) -> None:
        """Return partitioning factors based on current DVS."""

        self._THRESHOLD_N_FLAG = torch.where(self.kiosk.SURFACE_N > self.params.NTHRESH, 1.0, 0.0)
        self._THRESHOLD_N = torch.where(self.kiosk.SURFACE_N > self.params.NTHRESH, self.kiosk.SURFACE_N, 0.0)

        self._THRESHOLD_N_FLAG = torch.where(_emerging, 0.0, self._THRESHOLD_N_FLAG)
        self._THRESHOLD_N = torch.where(_emerging, 0.0, self._THRESHOLD_N)

    def integrate(self, day: date, delt: float = 1.0) -> None:
        """
        Update partitioning factors based on development stage (DVS)
        and the Nitrogen nutrition Index (NNI)
        """
        p = self.params
        s = self.states
        k = self.kiosk

        FRTMOD = torch.max(torch.tensor([1.0]).to(self.device), 1.0 / (k.RFTRA + 0.5).clamp(min=EPS))
        FLVMOD = torch.exp(-p.NPART * (1.0 - k.NNI))

        s.FR = torch.where(
            k.RFTRA < k.NNI,
            torch.min(torch.tensor([0.6]).to(self.device), p.FRTB(k.DVS) * FRTMOD),
            p.FRTB(k.DVS),
        )
        s.FL = torch.where(k.RFTRA < k.NNI, p.FLTB(k.DVS), p.FLTB(k.DVS) * FLVMOD)
        s.FS = torch.where(k.RFTRA < k.NNI, p.FSTB(k.DVS), p.FSTB(k.DVS) + p.FLTB(k.DVS) - s.FL)
        s.FO = p.FOTB(k.DVS)

        FSOMOD = 1 / torch.exp(-p.NPART * (1.0 - (self._THRESHOLD_N / p.NTHRESH).clamp(min=EPS)))
        s.FO = torch.where(self._THRESHOLD_N_FLAG.to(torch.bool), p.FOTB(k.DVS) * FSOMOD, s.FO)
        s.FL = torch.where(self._THRESHOLD_N_FLAG.to(torch.bool), p.FLTB(k.DVS) + p.FOTB(k.DVS) - s.FO, s.FL)
        s.FS = torch.where(self._THRESHOLD_N_FLAG.to(torch.bool), p.FSTB(k.DVS), s.FS)
        s.FR = torch.where(self._THRESHOLD_N_FLAG.to(torch.bool), p.FRTB(k.DVS), s.FR)

        s._update_kiosk()

    def reset(self, day: date, inds: torch.Tensor = None) -> None:
        """Reset states and rates"""
        inds = torch.ones((self.num_models,), device=self.device).to(torch.bool) if inds is None else inds

        k = self.kiosk
        s = self.states
        FR = self.params.FRTB(k.DVS)
        FL = self.params.FLTB(k.DVS)
        FS = self.params.FSTB(k.DVS)
        FO = self.params.FOTB(k.DVS)

        self._THRESHOLD_N_FLAG = torch.where(inds, 0.0, self._THRESHOLD_N_FLAG).detach()
        self._THRESHOLD_N = torch.where(inds, 0.0, self._THRESHOLD_N).detach()

        s.FR = torch.where(inds, FR, s.FR).detach()
        s.FL = torch.where(inds, FL, s.FL).detach()
        s.FS = torch.where(inds, FS, s.FS).detach()
        s.FO = torch.where(inds, FO, s.FO).detach()

        self.states._update_kiosk()

    def get_output(self, va: list[str] = None) -> torch.Tensor:
        """
        Return the output
        """
        if va is None:
            return self.states.FO
        else:
            output_vars = torch.empty(size=(self.num_models, len(va))).to(self.device)
            for i, v in enumerate(va):
                if v in self.states.trait_names():
                    output_vars[i, :] = getattr(self.states, v)
            return output_vars

    def get_extra_states(self) -> dict[str, torch.Tensor]:
        """
        Get extra states
        """
        return {"_THRESHOLD_N_FLAG", self._THRESHOLD_N_FLAG, "_THRESHOLD_N", self._THRESHOLD_N}

    def set_model_specific_params(self, k: str, v: torch.Tensor) -> None:
        """
        Set the specific parameters to handle overrides as needed
        Like casting to ints
        """
        setattr(self.params, k, v)
