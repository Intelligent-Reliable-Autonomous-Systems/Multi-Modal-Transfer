"""Calculates NPK Demand for the crop and corresponding uptake from soil

Written by: Will Solow, 2025
"""

from datetime import date
import torch

from collections import namedtuple
from model_engine.models.base_model import BatchTensorModel
from model_engine.models.states_rates import Tensor, NDArray, TensorBatchAfgenTrait
from model_engine.models.states_rates import (
    ParamTemplate,
    StatesTemplate,
    RatesTemplate,
    VariableKiosk,
)
from model_engine.inputs.input_providers import DFTensorWeatherDataContainer
from model_engine.util import EPS

MaxNutrientConcentrations = namedtuple(
    "MaxNutrientConcentrations",
    [
        "NMAXLV",
        "PMAXLV",
        "KMAXLV",
        "NMAXST",
        "PMAXST",
        "KMAXST",
        "NMAXRT",
        "PMAXRT",
        "KMAXRT",
        "NMAXSO",
        "PMAXSO",
        "KMAXSO",
    ],
)


class NPK_Demand_Uptake_TensorBatch(BatchTensorModel):
    """Calculates the crop N/P/K demand and its uptake from the soil."""

    class Parameters(ParamTemplate):
        NMAXLV_TB = TensorBatchAfgenTrait()
        PMAXLV_TB = TensorBatchAfgenTrait()
        KMAXLV_TB = TensorBatchAfgenTrait()

        NMAXRT_FR = Tensor(-99.0)
        PMAXRT_FR = Tensor(-99.0)
        KMAXRT_FR = Tensor(-99.0)

        NMAXST_FR = Tensor(-99.0)
        PMAXST_FR = Tensor(-99.0)
        KMAXST_FR = Tensor(-99.0)

        NMAXSO = Tensor(-99.0)
        PMAXSO = Tensor(-99.0)
        KMAXSO = Tensor(-99.0)

        TCNT = Tensor(-99.0)
        TCPT = Tensor(-99.0)
        TCKT = Tensor(-99.0)

        NFIX_FR = Tensor(-99.0)
        RNUPTAKEMAX = Tensor(-99.0)
        RPUPTAKEMAX = Tensor(-99.0)
        RKUPTAKEMAX = Tensor(-99.0)

        DVS_NPK_STOP = Tensor(-99.0)

    class RateVariables(RatesTemplate):
        RNUPTAKELV = Tensor(-99.0)
        RNUPTAKEST = Tensor(-99.0)
        RNUPTAKERT = Tensor(-99.0)
        RNUPTAKESO = Tensor(-99.0)

        RPUPTAKELV = Tensor(-99.0)
        RPUPTAKEST = Tensor(-99.0)
        RPUPTAKERT = Tensor(-99.0)
        RPUPTAKESO = Tensor(-99.0)

        RKUPTAKELV = Tensor(-99.0)
        RKUPTAKEST = Tensor(-99.0)
        RKUPTAKERT = Tensor(-99.0)
        RKUPTAKESO = Tensor(-99.0)

        RNUPTAKE = Tensor(-99.0)
        RPUPTAKE = Tensor(-99.0)
        RKUPTAKE = Tensor(-99.0)
        RNFIXATION = Tensor(-99.0)

        NDEMANDLV = Tensor(-99.0)
        NDEMANDST = Tensor(-99.0)
        NDEMANDRT = Tensor(-99.0)
        NDEMANDSO = Tensor(-99.0)

        PDEMANDLV = Tensor(-99.0)
        PDEMANDST = Tensor(-99.0)
        PDEMANDRT = Tensor(-99.0)
        PDEMANDSO = Tensor(-99.0)

        KDEMANDLV = Tensor(-99.0)
        KDEMANDST = Tensor(-99.0)
        KDEMANDRT = Tensor(-99.0)
        KDEMANDSO = Tensor(-99.0)

        NDEMAND = Tensor(-99.0)
        PDEMAND = Tensor(-99.0)
        KDEMAND = Tensor(-99.0)

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

        self.rates = self.RateVariables(
            num_models=self.num_models,
            kiosk=self.kiosk,
            publish=[
                "RNUPTAKE",
                "RPUPTAKE",
                "RKUPTAKE",
                "RNFIXATION",
                "RKFIXATION",
                "RPFIXATION",
                "RNUPTAKELV",
                "RNUPTAKEST",
                "RNUPTAKERT",
                "RNUPTAKESO",
                "RPUPTAKELV",
                "RPUPTAKEST",
                "RPUPTAKERT",
                "RPUPTAKESO",
                "RKUPTAKELV",
                "RKUPTAKEST",
                "RKUPTAKERT",
                "RKUPTAKESO",
            ],
        )

        self.zero_tens = torch.tensor([0.0]).to(self.device)

    def calc_rates(self, day: date, drv: DFTensorWeatherDataContainer, _emerging: torch.Tensor) -> None:
        """Calculate rates"""
        r = self.rates
        p = self.params
        k = self.kiosk

        delt = 1.0
        mc = self._compute_NPK_max_concentrations()

        r.NDEMANDLV = (
            torch.max(mc.NMAXLV * k.WLV - k.NAMOUNTLV, self.zero_tens)
            + torch.max(k.GRLV * mc.NMAXLV, self.zero_tens) * delt
        )
        r.NDEMANDST = (
            torch.max(mc.NMAXST * k.WST - k.NAMOUNTST, self.zero_tens)
            + torch.max(k.GRST * mc.NMAXST, self.zero_tens) * delt
        )
        r.NDEMANDRT = (
            torch.max(mc.NMAXRT * k.WRT - k.NAMOUNTRT, self.zero_tens)
            + torch.max(k.GRRT * mc.NMAXRT, self.zero_tens) * delt
        )
        r.NDEMANDSO = torch.max(mc.NMAXSO * k.WSO - k.NAMOUNTSO, self.zero_tens)

        r.PDEMANDLV = (
            torch.max(mc.PMAXLV * k.WLV - k.PAMOUNTLV, self.zero_tens)
            + torch.max(k.GRLV * mc.PMAXLV, self.zero_tens) * delt
        )
        r.PDEMANDST = (
            torch.max(mc.PMAXST * k.WST - k.PAMOUNTST, self.zero_tens)
            + torch.max(k.GRST * mc.PMAXST, self.zero_tens) * delt
        )
        r.PDEMANDRT = (
            torch.max(mc.PMAXRT * k.WRT - k.PAMOUNTRT, self.zero_tens)
            + torch.max(k.GRRT * mc.PMAXRT, self.zero_tens) * delt
        )
        r.PDEMANDSO = torch.max(mc.PMAXSO * k.WSO - k.PAMOUNTSO, self.zero_tens)

        r.KDEMANDLV = (
            torch.max(mc.KMAXLV * k.WLV - k.KAMOUNTLV, self.zero_tens)
            + torch.max(k.GRLV * mc.KMAXLV, self.zero_tens) * delt
        )
        r.KDEMANDST = (
            torch.max(mc.KMAXST * k.WST - k.KAMOUNTST, self.zero_tens)
            + torch.max(k.GRST * mc.KMAXST, self.zero_tens) * delt
        )
        r.KDEMANDRT = (
            torch.max(mc.KMAXRT * k.WRT - k.KAMOUNTRT, self.zero_tens)
            + torch.max(k.GRRT * mc.KMAXRT, self.zero_tens) * delt
        )
        r.KDEMANDSO = torch.max(mc.KMAXSO * k.WSO - k.KAMOUNTSO, self.zero_tens)

        r.NDEMAND = r.NDEMANDLV + r.NDEMANDST + r.NDEMANDRT
        r.PDEMAND = r.PDEMANDLV + r.PDEMANDST + r.PDEMANDRT
        r.KDEMAND = r.KDEMANDLV + r.KDEMANDST + r.KDEMANDRT

        r.RNUPTAKESO = torch.min(r.NDEMANDSO, k.NTRANSLOCATABLE) / (p.TCNT.clamp(min=EPS))
        r.RPUPTAKESO = torch.min(r.PDEMANDSO, k.PTRANSLOCATABLE) / (p.TCPT.clamp(min=EPS))
        r.RKUPTAKESO = torch.min(r.KDEMANDSO, k.KTRANSLOCATABLE) / (p.TCKT.clamp(min=EPS))

        NutrientLIMIT = torch.where(k.RFTRA > 0.01, 1.0, 0.0)

        r.RNFIXATION = torch.max(self.zero_tens, p.NFIX_FR * r.NDEMAND) * NutrientLIMIT

        r.RNUPTAKE = torch.where(
            k.DVS < p.DVS_NPK_STOP,
            (
                torch.max(
                    self.zero_tens,
                    torch.min(r.NDEMAND - r.RNFIXATION, torch.min(k.NAVAIL, p.RNUPTAKEMAX)),
                )
                * NutrientLIMIT
            ),
            0.0,
        )
        r.RPUPTAKE = torch.where(
            k.DVS < p.DVS_NPK_STOP,
            (torch.max(self.zero_tens, torch.min(r.PDEMAND, torch.min(k.PAVAIL, p.RPUPTAKEMAX))) * NutrientLIMIT),
            0.0,
        )
        r.RKUPTAKE = torch.where(
            k.DVS < p.DVS_NPK_STOP,
            (torch.max(self.zero_tens, torch.min(r.KDEMAND, torch.min(k.KAVAIL, p.RKUPTAKEMAX))) * NutrientLIMIT),
            0.0,
        )

        r.RNUPTAKELV = torch.where(
            r.NDEMAND == 0.0,
            0.0,
            (r.NDEMANDLV / (r.NDEMAND.clamp(min=EPS))) * (r.RNUPTAKE + r.RNFIXATION),
        )
        r.RNUPTAKEST = torch.where(
            r.NDEMAND == 0.0,
            0.0,
            (r.NDEMANDST / (r.NDEMAND.clamp(min=EPS))) * (r.RNUPTAKE + r.RNFIXATION),
        )
        r.RNUPTAKERT = torch.where(
            r.NDEMAND == 0.0,
            0.0,
            (r.NDEMANDRT / (r.NDEMAND.clamp(min=EPS))) * (r.RNUPTAKE + r.RNFIXATION),
        )

        r.RPUPTAKELV = torch.where(r.PDEMAND == 0.0, 0.0, (r.PDEMANDLV / (r.PDEMAND.clamp(min=EPS))) * r.RPUPTAKE)
        r.RPUPTAKEST = torch.where(r.PDEMAND == 0.0, 0.0, (r.PDEMANDST / (r.PDEMAND.clamp(min=EPS))) * r.RPUPTAKE)
        r.RPUPTAKERT = torch.where(r.PDEMAND == 0.0, 0.0, (r.PDEMANDRT / (r.PDEMAND.clamp(min=EPS))) * r.RPUPTAKE)

        r.RKUPTAKELV = torch.where(r.KDEMAND == 0.0, 0.0, (r.KDEMANDLV / (r.KDEMAND.clamp(min=EPS))) * r.RKUPTAKE)
        r.RKUPTAKEST = torch.where(r.KDEMAND == 0.0, 0.0, (r.KDEMANDST / (r.KDEMAND.clamp(min=EPS))) * r.RKUPTAKE)
        r.RKUPTAKERT = torch.where(r.KDEMAND == 0.0, 0.0, (r.KDEMANDRT / (r.KDEMAND.clamp(min=EPS))) * r.RKUPTAKE)

        # Set to 0 based on _emerging
        r.RNUPTAKELV = torch.where(_emerging, 0.0, r.RNUPTAKELV)
        r.RNUPTAKEST = torch.where(_emerging, 0.0, r.RNUPTAKEST)
        r.RNUPTAKERT = torch.where(_emerging, 0.0, r.RNUPTAKERT)
        r.RNUPTAKESO = torch.where(_emerging, 0.0, r.RNUPTAKESO)

        r.RPUPTAKELV = torch.where(_emerging, 0.0, r.RPUPTAKELV)
        r.RPUPTAKEST = torch.where(_emerging, 0.0, r.RPUPTAKEST)
        r.RPUPTAKERT = torch.where(_emerging, 0.0, r.RPUPTAKERT)
        r.RPUPTAKESO = torch.where(_emerging, 0.0, r.RPUPTAKESO)

        r.RKUPTAKELV = torch.where(_emerging, 0.0, r.RKUPTAKELV)
        r.RKUPTAKEST = torch.where(_emerging, 0.0, r.RKUPTAKEST)
        r.RKUPTAKERT = torch.where(_emerging, 0.0, r.RKUPTAKERT)
        r.RKUPTAKESO = torch.where(_emerging, 0.0, r.RKUPTAKESO)

        r.RNUPTAKE = torch.where(_emerging, 0.0, r.RNUPTAKE)
        r.RPUPTAKE = torch.where(_emerging, 0.0, r.RPUPTAKE)
        r.RKUPTAKE = torch.where(_emerging, 0.0, r.RKUPTAKE)
        r.RNFIXATION = torch.where(_emerging, 0.0, r.RNFIXATION)

        r.NDEMANDLV = torch.where(_emerging, 0.0, r.NDEMANDLV)
        r.NDEMANDST = torch.where(_emerging, 0.0, r.NDEMANDST)
        r.NDEMANDRT = torch.where(_emerging, 0.0, r.NDEMANDRT)
        r.NDEMANDSO = torch.where(_emerging, 0.0, r.NDEMANDSO)

        r.PDEMANDLV = torch.where(_emerging, 0.0, r.PDEMANDLV)
        r.PDEMANDST = torch.where(_emerging, 0.0, r.PDEMANDST)
        r.PDEMANDRT = torch.where(_emerging, 0.0, r.PDEMANDRT)
        r.PDEMANDSO = torch.where(_emerging, 0.0, r.PDEMANDSO)

        r.KDEMANDLV = torch.where(_emerging, 0.0, r.KDEMANDLV)
        r.KDEMANDST = torch.where(_emerging, 0.0, r.KDEMANDST)
        r.KDEMANDRT = torch.where(_emerging, 0.0, r.KDEMANDRT)
        r.KDEMANDSO = torch.where(_emerging, 0.0, r.KDEMANDSO)

        r.NDEMAND = torch.where(_emerging, 0.0, r.NDEMAND)
        r.PDEMAND = torch.where(_emerging, 0.0, r.PDEMAND)
        r.KDEMAND = torch.where(_emerging, 0.0, r.KDEMAND)

        self.rates._update_kiosk()

    def integrate(self, day: date, delt: float = 1.0) -> None:
        """Integrate states - no states to integrate in NPK Demand Uptake"""
        pass

    def _compute_NPK_max_concentrations(self) -> MaxNutrientConcentrations:
        """Computes the maximum N/P/K concentrations in leaves, stems, roots and storage organs.

        Note that max concentrations are first derived from the dilution curve for leaves.
        Maximum concentrations for stems and roots are computed as a fraction of the
        concentration for leaves. Maximum concentration for storage organs is directly taken from
        the parameters N/P/KMAXSO.
        """

        p = self.params
        k = self.kiosk
        NMAXLV = p.NMAXLV_TB(k.DVS)
        PMAXLV = p.PMAXLV_TB(k.DVS)
        KMAXLV = p.KMAXLV_TB(k.DVS)
        max_NPK_conc = MaxNutrientConcentrations(
            NMAXLV=NMAXLV,
            PMAXLV=PMAXLV,
            KMAXLV=KMAXLV,
            NMAXST=(p.NMAXST_FR * NMAXLV),
            NMAXRT=p.NMAXRT_FR * NMAXLV,
            NMAXSO=p.NMAXSO,
            PMAXST=p.PMAXST_FR * PMAXLV,
            PMAXRT=p.PMAXRT_FR * PMAXLV,
            PMAXSO=p.PMAXSO,
            KMAXST=p.KMAXST_FR * KMAXLV,
            KMAXRT=p.KMAXRT_FR * KMAXLV,
            KMAXSO=p.KMAXSO,
        )

        return max_NPK_conc

    def reset(self, day: date, inds: torch.Tensor = None) -> None:
        """Reset states and rates"""
        inds = torch.ones((self.num_models,), device=self.device).to(torch.bool) if inds is None else inds
        r = self.rates

        r.RNUPTAKELV = torch.where(inds, 0.0, r.RNUPTAKELV).detach()
        r.RNUPTAKEST = torch.where(inds, 0.0, r.RNUPTAKEST).detach()
        r.RNUPTAKERT = torch.where(inds, 0.0, r.RNUPTAKERT).detach()
        r.RNUPTAKESO = torch.where(inds, 0.0, r.RNUPTAKESO).detach()

        r.RPUPTAKELV = torch.where(inds, 0.0, r.RPUPTAKELV).detach()
        r.RPUPTAKEST = torch.where(inds, 0.0, r.RPUPTAKEST).detach()
        r.RPUPTAKERT = torch.where(inds, 0.0, r.RPUPTAKERT).detach()
        r.RPUPTAKESO = torch.where(inds, 0.0, r.RPUPTAKESO).detach()

        r.RKUPTAKELV = torch.where(inds, 0.0, r.RKUPTAKELV).detach()
        r.RKUPTAKEST = torch.where(inds, 0.0, r.RKUPTAKEST).detach()
        r.RKUPTAKERT = torch.where(inds, 0.0, r.RKUPTAKERT).detach()
        r.RKUPTAKESO = torch.where(inds, 0.0, r.RKUPTAKESO).detach()

        r.RNUPTAKE = torch.where(inds, 0.0, r.RNUPTAKE).detach()
        r.RPUPTAKE = torch.where(inds, 0.0, r.RPUPTAKE).detach()
        r.RKUPTAKE = torch.where(inds, 0.0, r.RKUPTAKE).detach()
        r.RNFIXATION = torch.where(inds, 0.0, r.RNFIXATION).detach()

        r.NDEMANDLV = torch.where(inds, 0.0, r.NDEMANDLV).detach()
        r.NDEMANDST = torch.where(inds, 0.0, r.NDEMANDST).detach()
        r.NDEMANDRT = torch.where(inds, 0.0, r.NDEMANDRT).detach()
        r.NDEMANDSO = torch.where(inds, 0.0, r.NDEMANDSO).detach()

        r.PDEMANDLV = torch.where(inds, 0.0, r.PDEMANDLV).detach()
        r.PDEMANDST = torch.where(inds, 0.0, r.PDEMANDST).detach()
        r.PDEMANDRT = torch.where(inds, 0.0, r.PDEMANDRT).detach()
        r.PDEMANDSO = torch.where(inds, 0.0, r.PDEMANDSO).detach()

        r.KDEMANDLV = torch.where(inds, 0.0, r.KDEMANDLV).detach()
        r.KDEMANDST = torch.where(inds, 0.0, r.KDEMANDST).detach()
        r.KDEMANDRT = torch.where(inds, 0.0, r.KDEMANDRT).detach()
        r.KDEMANDSO = torch.where(inds, 0.0, r.KDEMANDSO).detach()

        r.NDEMAND = torch.where(inds, 0.0, r.NDEMAND).detach()
        r.PDEMAND = torch.where(inds, 0.0, r.PDEMAND).detach()
        r.KDEMAND = torch.where(inds, 0.0, r.KDEMAND).detach()

        r._update_kiosk()

    def get_output(self, va: list[str] = None) -> torch.Tensor:
        """
        Return the output
        """
        if va is None:
            return self.rates.NDEMAND
        else:
            output_vars = torch.empty(size=(self.num_models, len(va))).to(self.device)
            for i, v in enumerate(va):
                if v in self.rates.trait_names():
                    output_vars[:, i] = getattr(self.rates, v)
            return output_vars

    def get_extra_states(self) -> dict[str, torch.Tensor]:
        """
        Get extra states
        """
        return {}

    def set_model_specific_params(self, k: str, v: torch.Tensor) -> None:
        """
        Set the specific parameters to handle overrides as needed
        Like casting to ints
        """
        setattr(self.params, k, v)
