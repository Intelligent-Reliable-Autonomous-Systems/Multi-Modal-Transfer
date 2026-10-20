"""Implementations of the WOFOST waterbalance modules for simulation
of NPK limited production

Written by: Will Solow, 2025
"""

from datetime import date

import torch
from model_engine.inputs.input_providers import DFTensorWeatherDataContainer
from model_engine.models.base_model import BatchTensorModel
from model_engine.models.states_rates import (
    ParamTemplate,
    RatesTemplate,
    StatesTemplate,
    Tensor,
    TensorBatchAfgenTrait,
    VariableKiosk,
)


class NPK_Soil_TensorBatch(BatchTensorModel):
    """A simple module for soil N/P/K dynamics."""

    _NSOILI = Tensor(-99.0)
    _PSOILI = Tensor(-99.0)
    _KSOILI = Tensor(-99.0)

    _FERT_N_SUPPLY = Tensor(0.0)
    _FERT_P_SUPPLY = Tensor(0.0)
    _FERT_K_SUPPLY = Tensor(0.0)

    class Parameters(ParamTemplate):
        NSOILBASE = Tensor(-99.0)
        NSOILBASE_FR = Tensor(-99.0)

        PSOILBASE = Tensor(-99.0)
        PSOILBASE_FR = Tensor(-99.0)

        KSOILBASE = Tensor(-99.0)
        KSOILBASE_FR = Tensor(-99.0)

        NAVAILI = Tensor(-99.0)
        PAVAILI = Tensor(-99.0)
        KAVAILI = Tensor(-99.0)

        NMAX = Tensor(-99.0)
        PMAX = Tensor(-99.0)
        KMAX = Tensor(-99.0)

        BG_N_SUPPLY = Tensor(-99.0)
        BG_P_SUPPLY = Tensor(-99.0)
        BG_K_SUPPLY = Tensor(-99.0)

        RNSOILMAX = Tensor(-99.0)
        RPSOILMAX = Tensor(-99.0)
        RKSOILMAX = Tensor(-99.0)

        RNABSORPTION = Tensor(-99.0)
        RPABSORPTION = Tensor(-99.0)
        RKABSORPTION = Tensor(-99.0)

        RNPKRUNOFF = TensorBatchAfgenTrait()

    class StateVariables(StatesTemplate):
        SURFACE_N = Tensor(-99.0)
        SURFACE_P = Tensor(-99.0)
        SURFACE_K = Tensor(-99.0)

        TOTN_RUNOFF = Tensor(-99.0)
        TOTP_RUNOFF = Tensor(-99.0)
        TOTK_RUNOFF = Tensor(-99.0)

        NSOIL = Tensor(-99.0)
        PSOIL = Tensor(-99.0)
        KSOIL = Tensor(-99.0)

        NAVAIL = Tensor(-99.0)
        PAVAIL = Tensor(-99.0)
        KAVAIL = Tensor(-99.0)

        TOTN = Tensor(-99.0)
        TOTP = Tensor(-99.0)
        TOTK = Tensor(-99.0)

    class RateVariables(RatesTemplate):
        RNSOIL = Tensor(-99.0)
        RPSOIL = Tensor(-99.0)
        RKSOIL = Tensor(-99.0)

        RNAVAIL = Tensor(-99.0)
        RPAVAIL = Tensor(-99.0)
        RKAVAIL = Tensor(-99.0)

        FERT_N_SUPPLY = Tensor(-99.0)
        FERT_P_SUPPLY = Tensor(-99.0)
        FERT_K_SUPPLY = Tensor(-99.0)

        RRUNOFF_N = Tensor(-99.0)
        RRUNOFF_P = Tensor(-99.0)
        RRUNOFF_K = Tensor(-99.0)

        RNSUBSOIL = Tensor(-99.0)
        RPSUBSOIL = Tensor(-99.0)
        RKSUBSOIL = Tensor(-99.0)

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

        p = self.params
        self.NSOILI = p.NSOILBASE
        self.PSOILI = p.PSOILBASE
        self.KSOILI = p.KSOILBASE

        self.states = self.StateVariables(
            num_models=self.num_models,
            kiosk=self.kiosk,
            publish=["SURFACE_N", "SURFACE_P", "SURFACE_K", "NAVAIL", "PAVAIL", "KAVAIL"],
            NSOIL=p.NSOILBASE,
            PSOIL=p.PSOILBASE,
            KSOIL=p.KSOILBASE,
            NAVAIL=p.NAVAILI,
            PAVAIL=p.PAVAILI,
            KAVAIL=p.KAVAILI,
            TOTN=0.0,
            TOTP=0.0,
            TOTK=0.0,
            SURFACE_N=0,
            SURFACE_P=0,
            SURFACE_K=0,
            TOTN_RUNOFF=0,
            TOTP_RUNOFF=0,
            TOTK_RUNOFF=0,
        )

        self.rates = self.RateVariables(num_models=self.num_models, kiosk=self.kiosk, publish=[])

        self.zero_tensor = torch.tensor([0.0]).to(self.device)

    def calc_rates(self, day: date, drv: DFTensorWeatherDataContainer) -> None:
        """Compute Rates for model"""
        r = self.rates
        s = self.states
        p = self.params
        k = self.kiosk

        r.FERT_N_SUPPLY = self._FERT_N_SUPPLY
        r.FERT_P_SUPPLY = self._FERT_P_SUPPLY
        r.FERT_K_SUPPLY = self._FERT_K_SUPPLY

        self._FERT_N_SUPPLY = torch.zeros((self.num_models,)).to(self.device)
        self._FERT_P_SUPPLY = torch.zeros((self.num_models,)).to(self.device)
        self._FERT_K_SUPPLY = torch.zeros((self.num_models,)).to(self.device)

        r.RRUNOFF_N = s.SURFACE_N * p.RNPKRUNOFF(k.DTSR)
        r.RRUNOFF_P = s.SURFACE_P * p.RNPKRUNOFF(k.DTSR)
        r.RRUNOFF_K = s.SURFACE_K * p.RNPKRUNOFF(k.DTSR)

        r.RNSUBSOIL = torch.min(p.RNSOILMAX, s.SURFACE_N * p.RNABSORPTION)
        r.RPSUBSOIL = torch.min(p.RPSOILMAX, s.SURFACE_P * p.RPABSORPTION)
        r.RKSUBSOIL = torch.min(p.RKSOILMAX, s.SURFACE_K * p.RKABSORPTION)

        r.RNSOIL = -torch.max(self.zero_tensor, torch.min(p.NSOILBASE_FR * self.NSOILI, s.NSOIL))
        r.RPSOIL = -torch.max(self.zero_tensor, torch.min(p.PSOILBASE_FR * self.PSOILI, s.PSOIL))
        r.RKSOIL = -torch.max(self.zero_tensor, torch.min(p.KSOILBASE_FR * self.KSOILI, s.KSOIL))

        RNUPTAKE = k.RNUPTAKE if "RNUPTAKE" in self.kiosk else torch.zeros((self.num_models,)).to(self.device)
        RPUPTAKE = k.RPUPTAKE if "RPUPTAKE" in self.kiosk else torch.zeros((self.num_models,)).to(self.device)
        RKUPTAKE = k.RKUPTAKE if "RKUPTAKE" in self.kiosk else torch.zeros((self.num_models,)).to(self.device)

        r.RNAVAIL = r.RNSUBSOIL + p.BG_N_SUPPLY - RNUPTAKE - r.RNSOIL
        r.RPAVAIL = r.RPSUBSOIL + p.BG_P_SUPPLY - RPUPTAKE - r.RPSOIL
        r.RKAVAIL = r.RKSUBSOIL + p.BG_K_SUPPLY - RKUPTAKE - r.RKSOIL

        self.rates._update_kiosk()

    def integrate(self, day: date, delt: float = 1.0) -> None:
        """Integrate states with rates"""
        r = self.rates
        s = self.states
        p = self.params

        s.SURFACE_N = s.SURFACE_N + (r.FERT_N_SUPPLY - r.RNSUBSOIL - r.RRUNOFF_N)
        s.SURFACE_P = s.SURFACE_P + (r.FERT_P_SUPPLY - r.RPSUBSOIL - r.RRUNOFF_P)
        s.SURFACE_K = s.SURFACE_K + (r.FERT_K_SUPPLY - r.RKSUBSOIL - r.RRUNOFF_K)

        s.TOTN_RUNOFF = s.TOTN_RUNOFF + r.RRUNOFF_N
        s.TOTP_RUNOFF = s.TOTP_RUNOFF + r.RRUNOFF_P
        s.TOTK_RUNOFF = s.TOTK_RUNOFF + r.RRUNOFF_K

        s.NSOIL = s.NSOIL + r.RNSOIL * delt
        s.PSOIL = s.PSOIL + r.RPSOIL * delt
        s.KSOIL = s.KSOIL + r.RKSOIL * delt

        s.NAVAIL = s.NAVAIL + r.RNAVAIL * delt
        s.PAVAIL = s.PAVAIL + r.RPAVAIL * delt
        s.KAVAIL = s.KAVAIL + r.RKAVAIL * delt

        s.NAVAIL = torch.min(s.NAVAIL, p.NMAX)
        s.PAVAIL = torch.min(s.PAVAIL, p.PMAX)
        s.KAVAIL = torch.min(s.KAVAIL, p.KMAX)

        self.states._update_kiosk()

    def _on_APPLY_NPK(
        self,
        N_amount: float,
        P_amount: float,
        K_amount: float,
        N_recovery: float,
        P_recovery: float,
        K_recovery: float,
    ) -> None:
        """Apply NPK based on amounts and update relevant parameters"""
        s = self.states
        if N_amount is not None:
            self._FERT_N_SUPPLY = N_amount * N_recovery
            s.TOTN = s.TOTN + N_amount
        if P_amount is not None:
            self._FERT_P_SUPPLY = P_amount * P_recovery
            s.TOTP = s.TOTP + P_amount
        if K_amount is not None:
            self._FERT_K_SUPPLY = K_amount * K_recovery
            s.TOTK = s.TOTK + K_amount

    def reset(self, day: date, inds: torch.Tensor = None) -> None:
        """Reset the model"""
        inds = torch.ones((self.num_models,), device=self.device).to(torch.bool) if inds is None else inds

        p = self.params
        s = self.states
        r = self.rates
        self.NSOILI = torch.where(inds, p.NSOILBASE, self.NSOILI).detach()
        self.PSOILI = torch.where(inds, p.PSOILBASE, self.PSOILI).detach()
        self.KSOILI = torch.where(inds, p.KSOILBASE, self.KSOILI).detach()

        s.SURFACE_N = torch.where(inds, 0.0, s.SURFACE_N).detach()
        s.SURFACE_P = torch.where(inds, 0.0, s.SURFACE_P).detach()
        s.SURFACE_K = torch.where(inds, 0.0, s.SURFACE_K).detach()

        s.TOTN_RUNOFF = torch.where(inds, 0.0, s.TOTN_RUNOFF).detach()
        s.TOTP_RUNOFF = torch.where(inds, 0.0, s.TOTP_RUNOFF).detach()
        s.TOTK_RUNOFF = torch.where(inds, 0.0, s.TOTK_RUNOFF).detach()

        s.NSOIL = torch.where(inds, p.NSOILBASE, s.NSOIL).detach()
        s.PSOIL = torch.where(inds, p.PSOILBASE, s.PSOIL).detach()
        s.KSOIL = torch.where(inds, p.KSOILBASE, s.KSOIL).detach()

        s.NAVAIL = torch.where(inds, p.NAVAILI, s.NAVAIL).detach()
        s.PAVAIL = torch.where(inds, p.PAVAILI, s.PAVAIL).detach()
        s.KAVAIL = torch.where(inds, p.KAVAILI, s.KAVAIL).detach()

        s.TOTN = torch.where(inds, 0.0, s.TOTN).detach()
        s.TOTP = torch.where(inds, 0.0, s.TOTP).detach()
        s.TOTK = torch.where(inds, 0.0, s.TOTK).detach()

        r.RNSOIL = torch.where(inds, 0.0, r.RNSOIL).detach()
        r.RPSOIL = torch.where(inds, 0.0, r.RPSOIL).detach()
        r.RKSOIL = torch.where(inds, 0.0, r.RKSOIL).detach()

        r.RNAVAIL = torch.where(inds, 0.0, r.RNAVAIL).detach()
        r.RPAVAIL = torch.where(inds, 0.0, r.RPAVAIL).detach()
        r.RKAVAIL = torch.where(inds, 0.0, r.RKAVAIL).detach()

        r.FERT_N_SUPPLY = torch.where(inds, 0.0, r.FERT_N_SUPPLY).detach()
        r.FERT_P_SUPPLY = torch.where(inds, 0.0, r.FERT_P_SUPPLY).detach()
        r.FERT_K_SUPPLY = torch.where(inds, 0.0, r.FERT_K_SUPPLY).detach()

        r.RRUNOFF_N = torch.where(inds, 0.0, r.RRUNOFF_N).detach()
        r.RRUNOFF_P = torch.where(inds, 0.0, r.RRUNOFF_P).detach()
        r.RRUNOFF_K = torch.where(inds, 0.0, r.RRUNOFF_K).detach()

        r.RNSUBSOIL = torch.where(inds, 0.0, r.RNSUBSOIL).detach()
        r.RPSUBSOIL = torch.where(inds, 0.0, r.RPSUBSOIL).detach()
        r.RKSUBSOIL = torch.where(inds, 0.0, r.RKSUBSOIL).detach()

        self.states._update_kiosk()
        self.rates._update_kiosk()

    def get_output(self, va: list[str] = None) -> torch.Tensor:
        """
        Return the output
        """
        if va is None:
            return self.states.NAVAIL
        else:
            output_vars = torch.empty(size=(self.num_models, len(va))).to(self.device)
            for i, v in enumerate(va):
                if v in self.states.trait_names():
                    output_vars[i, :] = getattr(self.states, v)
                elif v in self.rates.trait_names():
                    output_vars[i, :] = getattr(self.rates, v)
            return output_vars

    def set_model_specific_params(self, k: str, v: torch.Tensor) -> None:
        """
        Set the specific parameters to handle overrides as needed
        Like casting to ints
        """
        setattr(self.params, k, v)

    def get_extra_states(self) -> dict[str, torch.Tensor]:
        """
        Get extra states
        """
        return {
            "_NSOILI": self._NSOILI,
            "_PSOILI": self._PSOILI,
            "_KSOILI": self._KSOILI,
            "_FERT_N_SUPPLY": self._FERT_N_SUPPLY,
            "_FERT_P_SUPPLY": self._FERT_P_SUPPLY,
            "_FERT_K_SUPPLY": self._FERT_K_SUPPLY,
        }
