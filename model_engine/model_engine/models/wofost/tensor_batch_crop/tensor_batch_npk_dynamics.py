"""Overall implementation for the NPK dynamics of the crop including
subclasses to
    * NPK Demand Uptake
    * NPK Stress
    * NPK Translocation

Written by Will Solow, 2024
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
from model_engine.models.wofost.tensor_batch_crop.tensor_batch_nutrients.tensor_batch_npk_demand_uptake import (
    NPK_Demand_Uptake_TensorBatch as NPK_Demand_Uptake,
)
from model_engine.models.wofost.tensor_batch_crop.tensor_batch_nutrients.tensor_batch_npk_translocation import (
    NPK_Translocation_TensorBatch as NPK_Translocation,
)
from traitlets_pcse import Instance


class NPK_Crop_Dynamics_TensorBatch(BatchTensorModel):
    """Implementation of overall NPK crop dynamics."""

    translocation = Instance(BatchTensorModel)
    demand_uptake = Instance(BatchTensorModel)

    NAMOUNTLVI = Tensor(-99.0)
    NAMOUNTSTI = Tensor(-99.0)
    NAMOUNTRTI = Tensor(-99.0)
    NAMOUNTSOI = Tensor(-99.0)

    PAMOUNTLVI = Tensor(-99.0)
    PAMOUNTSTI = Tensor(-99.0)
    PAMOUNTRTI = Tensor(-99.0)
    PAMOUNTSOI = Tensor(-99.0)

    KAMOUNTLVI = Tensor(-99.0)
    KAMOUNTSTI = Tensor(-99.0)
    KAMOUNTRTI = Tensor(-99.0)
    KAMOUNTSOI = Tensor(-99.0)

    class Parameters(ParamTemplate):
        NMAXLV_TB = TensorBatchAfgenTrait()
        PMAXLV_TB = TensorBatchAfgenTrait()
        KMAXLV_TB = TensorBatchAfgenTrait()
        NMAXST_FR = Tensor(-99.0)
        NMAXRT_FR = Tensor(-99.0)
        PMAXST_FR = Tensor(-99.0)
        PMAXRT_FR = Tensor(-99.0)
        KMAXST_FR = Tensor(-99.0)
        KMAXRT_FR = Tensor(-99.0)
        NRESIDLV = Tensor(-99.0)
        NRESIDST = Tensor(-99.0)
        NRESIDRT = Tensor(-99.0)
        PRESIDLV = Tensor(-99.0)
        PRESIDST = Tensor(-99.0)
        PRESIDRT = Tensor(-99.0)
        KRESIDLV = Tensor(-99.0)
        KRESIDST = Tensor(-99.0)
        KRESIDRT = Tensor(-99.0)

    class StateVariables(StatesTemplate):
        NAMOUNTLV = Tensor(-99.0)
        PAMOUNTLV = Tensor(-99.0)
        KAMOUNTLV = Tensor(-99.0)

        NAMOUNTST = Tensor(-99.0)
        PAMOUNTST = Tensor(-99.0)
        KAMOUNTST = Tensor(-99.0)

        NAMOUNTSO = Tensor(-99.0)
        PAMOUNTSO = Tensor(-99.0)
        KAMOUNTSO = Tensor(-99.0)

        NAMOUNTRT = Tensor(-99.0)
        PAMOUNTRT = Tensor(-99.0)
        KAMOUNTRT = Tensor(-99.0)

        NUPTAKETOTAL = Tensor(-99.0)
        PUPTAKETOTAL = Tensor(-99.0)
        KUPTAKETOTAL = Tensor(-99.0)
        NFIXTOTAL = Tensor(-99.0)

        NlossesTotal = Tensor(-99.0)
        PlossesTotal = Tensor(-99.0)
        KlossesTotal = Tensor(-99.0)

    class RateVariables(RatesTemplate):
        RNAMOUNTLV = Tensor(-99.0)
        RPAMOUNTLV = Tensor(-99.0)
        RKAMOUNTLV = Tensor(-99.0)

        RNAMOUNTST = Tensor(-99.0)
        RPAMOUNTST = Tensor(-99.0)
        RKAMOUNTST = Tensor(-99.0)

        RNAMOUNTRT = Tensor(-99.0)
        RPAMOUNTRT = Tensor(-99.0)
        RKAMOUNTRT = Tensor(-99.0)

        RNAMOUNTSO = Tensor(-99.0)
        RPAMOUNTSO = Tensor(-99.0)
        RKAMOUNTSO = Tensor(-99.0)

        RNDEATHLV = Tensor(-99.0)
        RNDEATHST = Tensor(-99.0)
        RNDEATHRT = Tensor(-99.0)

        RPDEATHLV = Tensor(-99.0)
        RPDEATHST = Tensor(-99.0)
        RPDEATHRT = Tensor(-99.0)

        RKDEATHLV = Tensor(-99.0)
        RKDEATHST = Tensor(-99.0)
        RKDEATHRT = Tensor(-99.0)

        RNLOSS = Tensor(-99.0)
        RPLOSS = Tensor(-99.0)
        RKLOSS = Tensor(-99.0)

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

        self.translocation = NPK_Translocation(day, self.kiosk, parvalues, self.device, num_models=self.num_models)
        self.demand_uptake = NPK_Demand_Uptake(day, self.kiosk, parvalues, self.device, num_models=self.num_models)

        p = self.params
        k = self.kiosk

        self.NAMOUNTLVI = NAMOUNTLV = k.WLV * p.NMAXLV_TB(k.DVS)
        self.NAMOUNTSTI = NAMOUNTST = k.WST * p.NMAXLV_TB(k.DVS) * p.NMAXST_FR
        self.NAMOUNTRTI = NAMOUNTRT = k.WRT * p.NMAXLV_TB(k.DVS) * p.NMAXRT_FR
        self.NAMOUNTSOI = NAMOUNTSO = 0.0

        self.PAMOUNTLVI = PAMOUNTLV = k.WLV * p.PMAXLV_TB(k.DVS)
        self.PAMOUNTSTI = PAMOUNTST = k.WST * p.PMAXLV_TB(k.DVS) * p.PMAXST_FR
        self.PAMOUNTRTI = PAMOUNTRT = k.WRT * p.PMAXLV_TB(k.DVS) * p.PMAXRT_FR
        self.PAMOUNTSOI = PAMOUNTSO = 0.0

        self.KAMOUNTLVI = KAMOUNTLV = k.WLV * p.KMAXLV_TB(k.DVS)
        self.KAMOUNTSTI = KAMOUNTST = k.WST * p.KMAXLV_TB(k.DVS) * p.KMAXST_FR
        self.KAMOUNTRTI = KAMOUNTRT = k.WRT * p.KMAXLV_TB(k.DVS) * p.KMAXRT_FR
        self.KAMOUNTSOI = KAMOUNTSO = 0.0

        self.states = self.StateVariables(
            kiosk=self.kiosk,
            publish=[
                "NAMOUNTLV",
                "NAMOUNTST",
                "NAMOUNTRT",
                "PAMOUNTLV",
                "PAMOUNTST",
                "PAMOUNTRT",
                "KAMOUNTLV",
                "KAMOUNTST",
                "KAMOUNTRT",
                "NAMOUNTSO",
                "PAMOUNTSO",
                "KAMOUNTSO",
            ],
            NAMOUNTLV=NAMOUNTLV,
            NAMOUNTST=NAMOUNTST,
            NAMOUNTRT=NAMOUNTRT,
            NAMOUNTSO=NAMOUNTSO,
            PAMOUNTLV=PAMOUNTLV,
            PAMOUNTST=PAMOUNTST,
            PAMOUNTRT=PAMOUNTRT,
            PAMOUNTSO=PAMOUNTSO,
            KAMOUNTLV=KAMOUNTLV,
            KAMOUNTST=KAMOUNTST,
            KAMOUNTRT=KAMOUNTRT,
            KAMOUNTSO=KAMOUNTSO,
            NUPTAKETOTAL=0,
            PUPTAKETOTAL=0.0,
            KUPTAKETOTAL=0.0,
            NFIXTOTAL=0.0,
            NlossesTotal=0,
            PlossesTotal=0.0,
            KlossesTotal=0.0,
        )

        self.rates = self.RateVariables(kiosk=self.kiosk, publish=[])

    def calc_rates(self, day: date, drv: DFTensorWeatherDataContainer, _emerging: torch.Tensor) -> None:
        """Calculate state rates"""
        r = self.rates
        p = self.params
        k = self.kiosk

        self.demand_uptake.calc_rates(day, drv, _emerging)
        self.translocation.calc_rates(day, drv, _emerging)

        r.RNDEATHLV = p.NRESIDLV * k.DRLV
        r.RNDEATHST = p.NRESIDST * k.DRST
        r.RNDEATHRT = p.NRESIDRT * k.DRRT

        r.RPDEATHLV = p.PRESIDLV * k.DRLV
        r.RPDEATHST = p.PRESIDST * k.DRST
        r.RPDEATHRT = p.PRESIDRT * k.DRRT

        r.RKDEATHLV = p.KRESIDLV * k.DRLV
        r.RKDEATHST = p.KRESIDST * k.DRST
        r.RKDEATHRT = p.KRESIDRT * k.DRRT

        r.RNAMOUNTLV = k.RNUPTAKELV - k.RNTRANSLOCATIONLV - r.RNDEATHLV
        r.RNAMOUNTST = k.RNUPTAKEST - k.RNTRANSLOCATIONST - r.RNDEATHST
        r.RNAMOUNTRT = k.RNUPTAKERT - k.RNTRANSLOCATIONRT - r.RNDEATHRT
        r.RNAMOUNTSO = k.RNUPTAKESO

        r.RPAMOUNTLV = k.RPUPTAKELV - k.RPTRANSLOCATIONLV - r.RPDEATHLV
        r.RPAMOUNTST = k.RPUPTAKEST - k.RPTRANSLOCATIONST - r.RPDEATHST
        r.RPAMOUNTRT = k.RPUPTAKERT - k.RPTRANSLOCATIONRT - r.RPDEATHRT
        r.RPAMOUNTSO = k.RPUPTAKESO

        r.RKAMOUNTLV = k.RKUPTAKELV - k.RKTRANSLOCATIONLV - r.RKDEATHLV
        r.RKAMOUNTST = k.RKUPTAKEST - k.RKTRANSLOCATIONST - r.RKDEATHST
        r.RKAMOUNTRT = k.RKUPTAKERT - k.RKTRANSLOCATIONRT - r.RKDEATHRT
        r.RKAMOUNTSO = k.RKUPTAKESO

        r.RNLOSS = r.RNDEATHLV + r.RNDEATHST + r.RNDEATHRT
        r.RPLOSS = r.RPDEATHLV + r.RPDEATHST + r.RPDEATHRT
        r.RKLOSS = r.RKDEATHLV + r.RKDEATHST + r.RKDEATHRT

        # Evaluate to 0 when emerging
        r.RNAMOUNTLV = torch.where(_emerging, 0.0, r.RNAMOUNTLV)
        r.RPAMOUNTLV = torch.where(_emerging, 0.0, r.RPAMOUNTLV)
        r.RKAMOUNTLV = torch.where(_emerging, 0.0, r.RKAMOUNTLV)

        r.RNAMOUNTST = torch.where(_emerging, 0.0, r.RNAMOUNTST)
        r.RPAMOUNTST = torch.where(_emerging, 0.0, r.RPAMOUNTST)
        r.RKAMOUNTST = torch.where(_emerging, 0.0, r.RKAMOUNTST)

        r.RNAMOUNTRT = torch.where(_emerging, 0.0, r.RNAMOUNTRT)
        r.RPAMOUNTRT = torch.where(_emerging, 0.0, r.RPAMOUNTRT)
        r.RKAMOUNTRT = torch.where(_emerging, 0.0, r.RKAMOUNTRT)

        r.RNAMOUNTSO = torch.where(_emerging, 0.0, r.RNAMOUNTSO)
        r.RPAMOUNTSO = torch.where(_emerging, 0.0, r.RPAMOUNTSO)
        r.RKAMOUNTSO = torch.where(_emerging, 0.0, r.RKAMOUNTSO)

        r.RNDEATHLV = torch.where(_emerging, 0.0, r.RNDEATHLV)
        r.RNDEATHST = torch.where(_emerging, 0.0, r.RNDEATHST)
        r.RNDEATHRT = torch.where(_emerging, 0.0, r.RNDEATHRT)

        r.RPDEATHLV = torch.where(_emerging, 0.0, r.RPDEATHLV)
        r.RPDEATHST = torch.where(_emerging, 0.0, r.RPDEATHST)
        r.RPDEATHRT = torch.where(_emerging, 0.0, r.RPDEATHRT)

        r.RKDEATHLV = torch.where(_emerging, 0.0, r.RKDEATHLV)
        r.RKDEATHST = torch.where(_emerging, 0.0, r.RKDEATHST)
        r.RKDEATHRT = torch.where(_emerging, 0.0, r.RKDEATHRT)

        r.RNLOSS = torch.where(_emerging, 0.0, r.RNLOSS)
        r.RPLOSS = torch.where(_emerging, 0.0, r.RPLOSS)
        r.RKLOSS = torch.where(_emerging, 0.0, r.RKLOSS)

        self.rates._update_kiosk()

    def integrate(self, day: date, delt: float = 1.0) -> None:
        """Integrate state rates"""
        r = self.rates
        s = self.states
        k = self.kiosk

        s.NAMOUNTLV = s.NAMOUNTLV + r.RNAMOUNTLV
        s.NAMOUNTST = s.NAMOUNTST + r.RNAMOUNTST
        s.NAMOUNTRT = s.NAMOUNTRT + r.RNAMOUNTRT
        s.NAMOUNTSO = s.NAMOUNTSO + r.RNAMOUNTSO

        s.PAMOUNTLV = s.PAMOUNTLV + r.RPAMOUNTLV
        s.PAMOUNTST = s.PAMOUNTST + r.RPAMOUNTST
        s.PAMOUNTRT = s.PAMOUNTRT + r.RPAMOUNTRT
        s.PAMOUNTSO = s.PAMOUNTSO + r.RPAMOUNTSO

        s.KAMOUNTLV = s.KAMOUNTLV + r.RKAMOUNTLV
        s.KAMOUNTST = s.KAMOUNTST + r.RKAMOUNTST
        s.KAMOUNTRT = s.KAMOUNTRT + r.RKAMOUNTRT
        s.KAMOUNTSO = s.KAMOUNTSO + r.RKAMOUNTSO

        self.translocation.integrate(day, delt)
        self.demand_uptake.integrate(day, delt)

        s.NUPTAKETOTAL = s.NUPTAKETOTAL + k.RNUPTAKE
        s.PUPTAKETOTAL = s.PUPTAKETOTAL + k.RPUPTAKE
        s.KUPTAKETOTAL = s.KUPTAKETOTAL + k.RKUPTAKE
        s.NFIXTOTAL = s.NFIXTOTAL + k.RNFIXATION

        s.NlossesTotal = s.NlossesTotal + r.RNLOSS
        s.PlossesTotal = s.PlossesTotal + r.RPLOSS
        s.KlossesTotal = s.KlossesTotal + r.RKLOSS

        self.states._update_kiosk()

    def reset(self, day: date, inds: torch.Tensor = None) -> None:
        """Reset states and rates"""
        inds = torch.ones((self.num_models,), device=self.device).to(torch.bool) if inds is None else inds

        self.translocation.reset(day, inds=inds)
        self.demand_uptake.reset(day, inds=inds)

        p = self.params
        s = self.states
        r = self.rates
        k = self.kiosk

        self.NAMOUNTLVI = NAMOUNTLV = torch.where(inds, k.WLV * p.NMAXLV_TB(k.DVS), self.NAMOUNTLVI).detach()
        self.NAMOUNTSTI = NAMOUNTST = torch.where(
            inds, k.WST * p.NMAXLV_TB(k.DVS) * p.NMAXST_FR, self.NAMOUNTSTI
        ).detach()
        self.NAMOUNTRTI = NAMOUNTRT = torch.where(
            inds, k.WRT * p.NMAXLV_TB(k.DVS) * p.NMAXRT_FR, self.NAMOUNTRTI
        ).detach()
        self.NAMOUNTSOI = NAMOUNTSO = torch.where(inds, 0.0, self.NAMOUNTSOI).detach()

        self.PAMOUNTLVI = PAMOUNTLV = torch.where(inds, k.WLV * p.PMAXLV_TB(k.DVS), self.PAMOUNTLVI).detach()
        self.PAMOUNTSTI = PAMOUNTST = torch.where(
            inds, k.WST * p.PMAXLV_TB(k.DVS) * p.PMAXST_FR, self.PAMOUNTSTI
        ).detach()
        self.PAMOUNTRTI = PAMOUNTRT = torch.where(
            inds, k.WRT * p.PMAXLV_TB(k.DVS) * p.PMAXRT_FR, self.PAMOUNTRTI
        ).detach()
        self.PAMOUNTSOI = PAMOUNTSO = torch.where(inds, 0.0, self.PAMOUNTSOI).detach()

        self.KAMOUNTLVI = KAMOUNTLV = torch.where(inds, k.WLV * p.KMAXLV_TB(k.DVS), self.KAMOUNTLVI).detach()
        self.KAMOUNTSTI = KAMOUNTST = torch.where(
            inds, k.WST * p.KMAXLV_TB(k.DVS) * p.KMAXST_FR, self.KAMOUNTSTI
        ).detach()
        self.KAMOUNTRTI = KAMOUNTRT = torch.where(
            inds, k.WRT * p.KMAXLV_TB(k.DVS) * p.KMAXRT_FR, self.KAMOUNTRTI
        ).detach()
        self.KAMOUNTSOI = KAMOUNTSO = torch.where(inds, 0.0, self.KAMOUNTSOI).detach()

        s.NAMOUNTLV = torch.where(inds, NAMOUNTLV, s.NAMOUNTLV).detach()
        s.NAMOUNTST = torch.where(inds, NAMOUNTST, s.NAMOUNTST).detach()
        s.NAMOUNTRT = torch.where(inds, NAMOUNTRT, s.NAMOUNTRT).detach()
        s.NAMOUNTSO = torch.where(inds, NAMOUNTSO, s.NAMOUNTSO).detach()
        s.PAMOUNTLV = torch.where(inds, PAMOUNTLV, s.PAMOUNTLV).detach()
        s.PAMOUNTST = torch.where(inds, PAMOUNTST, s.PAMOUNTST).detach()
        s.PAMOUNTRT = torch.where(inds, PAMOUNTRT, s.PAMOUNTRT).detach()
        s.PAMOUNTSO = torch.where(inds, PAMOUNTSO, s.PAMOUNTSO).detach()
        s.KAMOUNTLV = torch.where(inds, KAMOUNTLV, s.KAMOUNTLV).detach()
        s.KAMOUNTST = torch.where(inds, KAMOUNTST, s.KAMOUNTST).detach()
        s.KAMOUNTRT = torch.where(inds, KAMOUNTRT, s.KAMOUNTRT).detach()
        s.KAMOUNTSO = torch.where(inds, KAMOUNTSO, s.KAMOUNTSO).detach()
        s.NUPTAKETOTAL = torch.where(inds, 0.0, s.NUPTAKETOTAL).detach()
        s.PUPTAKETOTAL = torch.where(inds, 0.0, s.PUPTAKETOTAL).detach()
        s.KUPTAKETOTAL = torch.where(inds, 0.0, s.KUPTAKETOTAL).detach()
        s.NFIXTOTAL = torch.where(inds, 0.0, s.NFIXTOTAL).detach()
        s.NlossesTotal = torch.where(inds, 0.0, s.NlossesTotal).detach()
        s.PlossesTotal = torch.where(inds, 0.0, s.PlossesTotal).detach()
        s.KlossesTotal = torch.where(inds, 0.0, s.KlossesTotal).detach()

        r.RNAMOUNTLV = torch.where(inds, 0.0, r.RNAMOUNTLV).detach()
        r.RPAMOUNTLV = torch.where(inds, 0.0, r.RPAMOUNTLV).detach()
        r.RKAMOUNTLV = torch.where(inds, 0.0, r.RKAMOUNTLV).detach()

        r.RNAMOUNTST = torch.where(inds, 0.0, r.RNAMOUNTST).detach()
        r.RPAMOUNTST = torch.where(inds, 0.0, r.RPAMOUNTST).detach()
        r.RKAMOUNTST = torch.where(inds, 0.0, r.RKAMOUNTST).detach()

        r.RNAMOUNTRT = torch.where(inds, 0.0, r.RNAMOUNTRT).detach()
        r.RPAMOUNTRT = torch.where(inds, 0.0, r.RPAMOUNTRT).detach()
        r.RKAMOUNTRT = torch.where(inds, 0.0, r.RKAMOUNTRT).detach()

        r.RNAMOUNTSO = torch.where(inds, 0.0, r.RNAMOUNTSO).detach()
        r.RPAMOUNTSO = torch.where(inds, 0.0, r.RPAMOUNTSO).detach()
        r.RKAMOUNTSO = torch.where(inds, 0.0, r.RKAMOUNTSO).detach()

        r.RNDEATHLV = torch.where(inds, 0.0, r.RNDEATHLV).detach()
        r.RNDEATHST = torch.where(inds, 0.0, r.RNDEATHST).detach()
        r.RNDEATHRT = torch.where(inds, 0.0, r.RNDEATHRT).detach()

        r.RPDEATHLV = torch.where(inds, 0.0, r.RPDEATHLV).detach()
        r.RPDEATHST = torch.where(inds, 0.0, r.RPDEATHST).detach()
        r.RPDEATHRT = torch.where(inds, 0.0, r.RPDEATHRT).detach()

        r.RKDEATHLV = torch.where(inds, 0.0, r.RKDEATHLV).detach()
        r.RKDEATHST = torch.where(inds, 0.0, r.RKDEATHST).detach()
        r.RKDEATHRT = torch.where(inds, 0.0, r.RKDEATHRT).detach()

        r.RNLOSS = torch.where(inds, 0.0, r.RNLOSS).detach()
        r.RPLOSS = torch.where(inds, 0.0, r.RPLOSS).detach()
        r.RKLOSS = torch.where(inds, 0.0, r.RKLOSS).detach()

        self.states._update_kiosk()
        self.rates._update_kiosk()

    def get_output(self, va: list[str] = None) -> torch.Tensor:
        """
        Return the output
        """
        if va is None:
            return self.states.NUPTAKETOTAL
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
        return {
            "NAMOUNTLVI",
            self.NAMOUNTLVI,
            "NAMOUNTSTI",
            self.NAMOUNTSTI,
            "NAMOUNTRTI",
            self.NAMOUNTRTI,
            "NAMOUNTSOI",
            self.NAMOUNTSOI,
            "PAMOUNTLVI",
            self.PAMOUNTLVI,
            "PAMOUNTSTI",
            self.PAMOUNTSTI,
            "PAMOUNTRTI",
            self.PAMOUNTRTI,
            "PAMOUNTSOI",
            self.PAMOUNTSOI,
            "KAMOUNTLVI",
            self.KAMOUNTLVI,
            "KAMOUNTSTI",
            self.KAMOUNTSTI,
            "KAMOUNTRTI",
            self.KAMOUNTRTI,
            "KAMOUNTSOI",
            self.KAMOUNTSOI,
        }

    def set_model_specific_params(self, k: str, v: torch.Tensor) -> None:
        """
        Set the specific parameters to handle overrides as needed
        Like casting to ints
        """
        setattr(self.params, k, v)
