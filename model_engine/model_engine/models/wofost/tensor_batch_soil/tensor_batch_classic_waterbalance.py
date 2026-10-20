"""Python implementations of the WOFOST waterbalance modules for simulation
of and water-limited production under freely draining conditions.

Written by: Will Solow, 2025
"""

from datetime import date

import numpy as np
import torch
from model_engine.inputs.input_providers import DFTensorWeatherDataContainer
from model_engine.models.base_model import BatchTensorModel
from model_engine.models.states_rates import (
    ParamTemplate,
    RatesTemplate,
    StatesTemplate,
    Tensor,
    TensorBatchAfgen,
    VariableKiosk,
)
from model_engine.util import EPS
from traitlets_pcse import Instance, List


class WaterbalanceFD_TensorBatch(BatchTensorModel):
    """Waterbalance for freely draining soils under water-limited production."""

    RDold = Tensor(-99.0)
    RDM = Tensor(-99.0)

    DSLR = Tensor(-99.0)

    RINold = Tensor(-99)

    NINFTB = Instance(TensorBatchAfgen)

    _RIRR = Tensor(0.0)

    DEFAULT_RD = Tensor(10.0)

    _increments_W = List()

    class Parameters(ParamTemplate):
        SMFCF = Tensor(-99.0)
        SM0 = Tensor(-99.0)
        SMW = Tensor(-99.0)
        CRAIRC = Tensor(-99.0)
        SOPE = Tensor(-99.0)
        KSUB = Tensor(-99.0)
        RDMSOL = Tensor(-99.0)
        SMLIM = Tensor(-99.0)

        IFUNRN = Tensor(-99.0)
        SSMAX = Tensor(-99.0)
        SSI = Tensor(-99.0)
        WAV = Tensor(-99.0)
        NOTINF = Tensor(-99.0)

    class StateVariables(StatesTemplate):
        SM = Tensor(-99.0)
        SS = Tensor(-99.0)
        SSI = Tensor(-99.0)
        WC = Tensor(-99.0)
        WI = Tensor(-99.0)
        WLOW = Tensor(-99.0)
        WLOWI = Tensor(-99.0)
        WWLOW = Tensor(-99.0)

        WTRAT = Tensor(-99.0)
        EVST = Tensor(-99.0)
        EVWT = Tensor(-99.0)
        TSR = Tensor(-99.0)
        RAINT = Tensor(-99.0)
        WART = Tensor(-99.0)
        TOTINF = Tensor(-99.0)
        TOTIRR = Tensor(-99.0)
        TOTIRRIG = Tensor(-99.0)
        PERCT = Tensor(-99.0)
        LOSST = Tensor(-99.0)

        WBALRT = Tensor(-99.0)
        WBALTT = Tensor(-99.0)
        DSOS = Tensor(-99)

    class RateVariables(RatesTemplate):
        EVS = Tensor(-99.0)
        EVW = Tensor(-99.0)
        WTRA = Tensor(-99.0)
        RIN = Tensor(-99.0)
        RIRR = Tensor(-99.0)
        PERC = Tensor(-99.0)
        LOSS = Tensor(-99.0)
        DW = Tensor(-99.0)
        DWLOW = Tensor(-99.0)
        DTSR = Tensor(-99.0)
        DSS = Tensor(-99.0)
        DRAINT = Tensor(-99.0)

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

        p.SM0 = torch.where(p.SM0 < p.SMW, p.SMW + 0.000001, p.SM0)

        SMLIM = torch.clamp(p.SMLIM, p.SMW, p.SM0)

        RD = torch.tile(self.DEFAULT_RD, (self.num_models,))
        RDM = torch.max(RD, p.RDMSOL)
        self.RDold = RD
        self.RDM = RDM
        SS = p.SSI

        SM = torch.clamp((p.SMW + p.WAV / (RD.clamp(min=EPS))), p.SMW, SMLIM)
        WC = SM * RD
        WI = WC

        WLOW = torch.clamp(
            (p.WAV + RDM * p.SMW - WC),
            torch.tensor([0.0]).to(self.device),
            p.SM0 * (RDM - RD),
        )
        WLOWI = WLOW
        WWLOW = WC + WLOW

        self.DSLR = torch.where((SM >= (p.SMW + 0.5 * (p.SMFCF - p.SMW))), 1.0, 5.0)

        self.RINold = 0.0
        NINFTB = np.tile([0.0, 0.0, 0.5, 0.0, 1.5, 1.0], self.num_models).astype(np.float32)
        self.NINFTB = TensorBatchAfgen(np.reshape(NINFTB, (self.num_models, -1)).tolist())
        self.states = self.StateVariables(
            num_models=self.num_models,
            kiosk=self.kiosk,
            publish=["SM", "DSOS"],
            SM=SM,
            SS=SS,
            SSI=p.SSI,
            WC=WC,
            WI=WI,
            WLOW=WLOW,
            WLOWI=WLOWI,
            WWLOW=WWLOW,
            WTRAT=0.0,
            EVST=0.0,
            EVWT=0.0,
            TSR=0.0,
            RAINT=0.0,
            WART=0.0,
            TOTINF=0.0,
            TOTIRR=0.0,
            DSOS=0,
            PERCT=0.0,
            LOSST=0.0,
            WBALRT=-999.0,
            WBALTT=-999.0,
            TOTIRRIG=0.0,
        )
        self.rates = self.RateVariables(num_models=self.num_models, kiosk=self.kiosk, publish=["DTSR", "EVS"])

        self._increments_W = []

        self.zero_tensor = torch.tensor([0]).to(self.device)

    def calc_rates(self, day: date, drv: DFTensorWeatherDataContainer) -> None:
        """Calculate state rates for integration"""
        s = self.states
        p = self.params
        r = self.rates
        k = self.kiosk

        r.RIRR = self._RIRR
        self._RIRR = 0.0

        if "TRA" not in self.kiosk:
            r.WTRA = torch.zeros(self.num_models).to(self.device)
            EVWMX = torch.tensor(drv.E0).to(self.device)
            EVSMX = torch.tensor(drv.ES0).to(self.device)
        else:
            r.WTRA = k.TRA
            EVWMX = k.EVWMX
            EVSMX = k.EVSMX

        r.EVW = torch.where(s.SS > 1, EVWMX, 0)

        EVSMXT = EVSMX * (torch.sqrt(self.DSLR + 1) - torch.sqrt(self.DSLR))
        r.EVS = torch.where(
            s.SS > 1,
            0,
            torch.where(self.RINold >= 1, EVSMX, torch.min(EVSMX, EVSMXT + self.RINold)),
        )

        self.DSLR = torch.where(s.SS > 1, self.DSLR, torch.where(self.RINold >= 1, 1, self.DSLR + 1))

        RINPRE = torch.where(
            p.IFUNRN == 0,
            (1.0 - p.NOTINF) * drv.RAIN,
            (1.0 - p.NOTINF * self.NINFTB(drv.RAIN)) * drv.RAIN,
        )
        RINPRE = RINPRE + r.RIRR + s.SS

        RINPRE = torch.where(s.SS > 0.1, torch.min(p.SOPE, RINPRE + r.RIRR - r.EVW), RINPRE)

        RD = self._determine_rooting_depth()
        WE = p.SMFCF * RD

        PERC1 = torch.clamp((s.WC - WE) - r.WTRA - r.EVS, self.zero_tensor, p.SOPE)

        WELOW = p.SMFCF * (self.RDM - RD)
        r.LOSS = torch.clamp((s.WLOW - WELOW + PERC1), self.zero_tensor, p.KSUB)

        PERC2 = ((self.RDM - RD) * p.SM0 - s.WLOW) + r.LOSS
        r.PERC = torch.min(PERC1, PERC2)

        r.RIN = torch.min(RINPRE, (p.SM0 - s.SM) * RD + r.WTRA + r.EVS + r.PERC)
        self.RINold = r.RIN

        r.DW = r.RIN - r.WTRA - r.EVS - r.PERC
        r.DWLOW = r.PERC - r.LOSS

        Wtmp = s.WC + r.DW

        r.EVS = torch.where(Wtmp < 0.0, r.EVS + Wtmp, r.EVS)
        r.DW = torch.where(Wtmp < 0.0, -s.WC, r.DW)

        SStmp = drv.RAIN + r.RIRR - r.EVW - r.RIN
        r.DSS = torch.min(SStmp, (p.SSMAX - s.SS))
        r.DTSR = SStmp - r.DSS
        r.DRAINT = drv.RAIN

        self.rates._update_kiosk()

    def integrate(self, day: date, delt: float = 1.0) -> None:
        """Integrate states from rates"""
        s = self.states
        p = self.params
        r = self.rates

        s.WTRAT = s.WTRAT + r.WTRA * delt
        s.EVWT = s.EVWT + r.EVW * delt
        s.EVST = s.EVST + r.EVS * delt

        s.RAINT = s.RAINT + r.DRAINT * delt
        s.TOTINF = s.TOTINF + r.RIN * delt
        s.TOTIRR = s.TOTIRR + r.RIRR * delt

        s.SS = s.SS + r.DSS * delt
        s.TSR = s.TSR + r.DTSR * delt

        s.WC = s.WC + r.DW * delt
        assert torch.any(s.WC >= 0.0), "Negative amount of water in root zone on day %s: %s" % (
            day,
            s.WC,
        )

        s.PERCT = s.PERCT + r.PERC * delt
        s.LOSST = s.LOSST + r.LOSS * delt

        s.WLOW = s.WLOW + r.DWLOW * delt
        s.WWLOW = s.WC + s.WLOW * delt

        RD = self._determine_rooting_depth()
        RDchange = RD - self.RDold
        self._redistribute_water(RDchange)

        s.SM = s.WC / (RD.clamp(min=EPS))

        s.DSOS = torch.where(s.SM >= (p.SM0 - p.CRAIRC), s.DSOS + 1, s.DSOS)

        self.RDold = RD

        self.states._update_kiosk()

    def _determine_rooting_depth(self) -> torch.Tensor:
        """Determines appropriate use of the rooting depth (RD)"""
        if "RD" in self.kiosk:
            return self.kiosk.RD
        else:
            return self.DEFAULT_RD

    def _redistribute_water(self, RDchange: torch.Tensor) -> None:
        """Redistributes the water between the root zone and the lower zone."""
        s = self.states
        p = self.params

        WDR = torch.zeros((self.num_models,)).to(self.device)

        WDR = torch.where(
            RDchange > 0.001,
            torch.min(s.WLOW, s.WLOW * RDchange / (p.RDMSOL - self.RDold).clamp(min=EPS)),
            s.WC * RDchange / (self.RDold.clamp(min=EPS)),
        )
        s.WLOW = torch.where(WDR != 0.0, s.WLOW - WDR, s.WLOW)
        s.WC = torch.where(WDR != 0.0, s.WC + WDR, s.WC)
        s.WART = torch.where(WDR != 0.0, s.WART + WDR, s.WART)

    def reset(self, day: date, inds: torch.Tensor = None) -> None:
        """Reset the model"""
        inds = torch.ones((self.num_models,), device=self.device).to(torch.bool) if inds is None else inds

        p = self.params
        s = self.states
        r = self.rates

        p.SM0 = torch.where(p.SM0 < p.SMW, p.SMW + 0.000001, p.SM0)

        SMLIM = torch.clamp(p.SMLIM, p.SMW, p.SM0)

        RD = self.DEFAULT_RD
        RDM = torch.max(RD, p.RDMSOL)
        self.RDold = torch.where(inds, RD, self.RDold).detach()
        self.RDM = torch.where(inds, RDM, self.RDM).detach()

        SS = p.SSI
        SM = torch.clamp((p.SMW + p.WAV / (RD.clamp(min=EPS))), p.SMW, SMLIM)
        WC = SM * RD
        WI = WC

        WLOW = torch.clamp(
            (p.WAV + RDM * p.SMW - WC),
            torch.tensor([0.0]).to(self.device),
            p.SM0 * (RDM - RD),
        )
        WLOWI = WLOW
        WWLOW = WC + WLOW

        self.DSLR = torch.where(
            inds, torch.where((SM >= (p.SMW + 0.5 * (p.SMFCF - p.SMW))), 1.0, 5.0), self.DSLR
        ).detach()

        self.RINold = 0.0
        NINFTB = np.tile([0.0, 0.0, 0.5, 0.0, 1.5, 1.0], self.num_models).astype(np.float32)
        self.NINFTB = TensorBatchAfgen(np.reshape(NINFTB, (self.num_models, -1)).tolist())

        s.SM = torch.where(inds, SM, s.SM).detach()
        s.SS = torch.where(inds, SS, s.SS).detach()
        s.SSI = torch.where(inds, p.SSI, s.SSI).detach()
        s.WC = torch.where(inds, WC, s.WC).detach()
        s.WI = torch.where(inds, WI, s.WI).detach()
        s.WLOW = torch.where(inds, WLOW, s.WLOW).detach()
        s.WWLOW = torch.where(inds, WWLOW, s.WWLOW).detach()
        s.WLOWI = torch.where(inds, WLOWI, s.WLOWI).detach()
        s.WTRAT = torch.where(inds, 0.0, s.WTRAT).detach()
        s.EVST = torch.where(inds, 0.0, s.EVST).detach()
        s.EVWT = torch.where(inds, 0.0, s.EVWT).detach()
        s.TSR = torch.where(inds, 0.0, s.TSR).detach()
        s.RAINT = torch.where(inds, 0.0, s.RAINT).detach()
        s.WART = torch.where(inds, 0.0, s.WART).detach()
        s.TOTINF = torch.where(inds, 0.0, s.TOTINF).detach()
        s.TOTIRR = torch.where(inds, 0.0, s.TOTIRR).detach()
        s.DSOS = torch.where(inds, 0.0, s.DSOS).detach()
        s.PERCT = torch.where(inds, 0.0, s.PERCT).detach()
        s.LOSST = torch.where(inds, 0.0, s.LOSST).detach()
        s.WBALRT = torch.where(inds, 999.0, s.WBALRT).detach()
        s.WBALTT = torch.where(inds, -999.0, s.WBALTT).detach()
        s.TOTIRRIG = torch.where(inds, 0.0, s.TOTIRRIG).detach()

        r.EVS = torch.where(inds, 0.0, r.EVS).detach()
        r.EVW = torch.where(inds, 0.0, r.EVW).detach()
        r.WTRA = torch.where(inds, 0.0, r.WTRA).detach()
        r.RIN = torch.where(inds, 0.0, r.RIN).detach()
        r.RIRR = torch.where(inds, 0.0, r.RIRR).detach()
        r.PERC = torch.where(inds, 0.0, r.PERC).detach()
        r.LOSS = torch.where(inds, 0.0, r.LOSS).detach()
        r.DW = torch.where(inds, 0.0, r.DW).detach()
        r.DWLOW = torch.where(inds, 0.0, r.DWLOW).detach()
        r.DTSR = torch.where(inds, 0.0, r.DTSR).detach()
        r.DSS = torch.where(inds, 0.0, r.DSS).detach()
        r.DRAINT = torch.where(inds, 0.0, r.DRAINT).detach()

        self.states._update_kiosk()
        self.rates._update_kiosk()

        self._increments_W = []

    def get_output(self, va: list[str] = None) -> torch.Tensor:
        """
        Return the output
        """
        if va is None:
            return self.states.SM
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
            "RDold": self.RDold,
            "RDM": self.RDM,
            "DSLR": self.DSLR,
            "RINold": self.RINold,
            "NINFTB": self.NINFTB,
            "_RIRR": self._RIRR,
            "DEFAULT_RD": self.DEFAULT_RD,
            "_increments_W": self._increments_W,
        }
