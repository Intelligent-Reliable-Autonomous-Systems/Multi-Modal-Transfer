"""Handles growth of leaf dynamics in the crop

Written by: Allard de Wit (allard.dewit@wur.nl), April 2014
Modified by Will Solow, 2024
"""

from datetime import date

import torch
from model_engine.models.base_model import BatchTensorModel
from model_engine.models.states_rates import (
    ParamTemplate,
    RatesTemplate,
    StatesTemplate,
    Tensor,
    TensorBatchAfgenTrait,
    VariableKiosk,
)
from model_engine.util import EPS, tensor_appendleft


class WOFOST_Leaf_Dynamics_NPK_TensorBatch(BatchTensorModel):
    """Leaf dynamics for the WOFOST crop model including leaf response to
    NPK stress.
    """

    LV = Tensor(-99.0)
    SLA = Tensor(-99.0)
    LVAGE = Tensor(-99.0)
    LVPOINTER = Tensor(-99.0)
    MAX_LEN = 365

    class Parameters(ParamTemplate):
        RGRLAI = Tensor(-99.0)
        SPAN = Tensor(-99.0)
        TBASE = Tensor(-99.0)
        PERDL = Tensor(-99.0)
        TDWI = Tensor(-99.0)
        SLATB = TensorBatchAfgenTrait()
        KDIFTB = TensorBatchAfgenTrait()
        RDRLV_NPK = Tensor(-99.0)
        NSLA_NPK = Tensor(-99.0)
        NLAI_NPK = Tensor(-99.0)

    class StateVariables(StatesTemplate):
        LAIEM = Tensor(-99.0)
        LASUM = Tensor(-99.0)
        LAIEXP = Tensor(-99.0)
        LAIMAX = Tensor(-99.0)
        LAI = Tensor(-99.0)
        WLV = Tensor(-99.0)
        DWLV = Tensor(-99.0)
        TWLV = Tensor(-99.0)

    class RateVariables(RatesTemplate):
        GRLV = Tensor(-99.0)
        DSLV1 = Tensor(-99.0)
        DSLV2 = Tensor(-99.0)
        DSLV3 = Tensor(-99.0)
        DSLV4 = Tensor(-99.0)
        DSLV = Tensor(-99.0)
        DALV = Tensor(-99.0)
        DRLV = Tensor(-99.0)
        SLAT = Tensor(-99.0)
        FYSAGE = Tensor(-99.0)
        GLAIEX = Tensor(-99.0)
        GLASOL = Tensor(-99.0)

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
        k = self.kiosk

        WLV = (p.TDWI * (1 - k.FR)) * k.FL
        DWLV = 0.0
        TWLV = WLV + DWLV

        SLA = torch.zeros((self.num_models, self.MAX_LEN)).to(self.device)
        LVAGE = torch.zeros((self.num_models, self.MAX_LEN)).to(self.device)
        LV = torch.zeros((self.num_models, self.MAX_LEN)).to(self.device)
        SLA[:, 0] = p.SLATB(k.DVS)
        LV[:, 0] = WLV

        LAIEM = LV[:, 0] * SLA[:, 0]
        LASUM = LAIEM
        LAIEXP = LAIEM
        LAIMAX = LAIEM
        SAI = PAI = 0
        if "SAI" in self.kiosk:
            SAI = k.SAI
        if "PAI" in self.kiosk:
            PAI = k.PAI
        LAI = LASUM + SAI + PAI

        self.LV = LV
        self.SLA = SLA
        self.LVAGE = LVAGE
        self.LVPOINTER = torch.ones((self.num_models,)).to(self.device)

        self.states = self.StateVariables(
            num_models=self.num_models,
            kiosk=self.kiosk,
            publish=["LAI", "WLV", "TWLV"],
            LAIEM=LAIEM,
            LASUM=LASUM,
            LAIEXP=LAIEXP,
            LAIMAX=LAIMAX,
            LAI=LAI,
            WLV=WLV,
            DWLV=DWLV,
            TWLV=TWLV,
        )

        self.rates = self.RateVariables(num_models=self.num_models, kiosk=self.kiosk, publish=["GRLV", "DRLV"])

    def _calc_LAI(self) -> torch.Tensor:
        """Compute LAI as Total leaf area Index as sum of leaf, pod and stem area"""
        k = self.kiosk
        SAI = PAI = 0
        if "SAI" in self.kiosk:
            SAI = k.SAI
        if "PAI" in self.kiosk:
            PAI = k.PAI
        return self.states.LASUM + SAI + PAI

    def calc_rates(self, day: date, drv, _emerging: torch.Tensor) -> None:
        """Calculate state rates"""
        r = self.rates
        s = self.states
        p = self.params
        k = self.kiosk

        r.GRLV = k.ADMI * k.FL
        r.DSLV1 = s.WLV * (1.0 - k.RFTRA) * p.PERDL

        LAICR = 3.2 / p.KDIFTB(k.DVS).clamp(min=EPS)

        r.DSLV2 = s.WLV * torch.clamp(
            0.03 * (s.LAI - LAICR) / LAICR.clamp(min=EPS),
            torch.tensor([0.0]).to(self.device),
            torch.tensor([0.03]).to(self.device),
        )

        if "RF_FROST" in self.kiosk:
            r.DSLV3 = s.WLV * k.RF_FROST
        else:
            r.DSLV3 = torch.zeros((self.num_models,)).to(self.device)

        r.DSLV4 = s.WLV * p.RDRLV_NPK * (1.0 - k.NPKI)
        r.DSLV = torch.max(torch.max(r.DSLV1, r.DSLV2), r.DSLV3) + r.DSLV4

        DALV = torch.where(self.LVAGE > p.SPAN.unsqueeze(1), self.LV, 0.0)
        r.DALV = torch.sum(DALV, dim=1)
        r.DRLV = torch.max(r.DSLV, r.DALV)

        r.FYSAGE = torch.max(torch.tensor([0.0]).to(self.device), (drv.TEMP - p.TBASE) / (35.0 - p.TBASE))
        sla_npk_factor = torch.exp(-p.NSLA_NPK * (1.0 - k.NPKI))
        r.SLAT = p.SLATB(k.DVS) * sla_npk_factor

        factor = torch.where((k.DVS < 0.2) & (s.LAI < 0.75), k.RFTRA * torch.exp(-p.NLAI_NPK * (1.0 - k.NPKI)), 1.0)
        DTEFF = torch.max(torch.tensor([0.0]).to(self.device), drv.TEMP - p.TBASE)

        r.GLAIEX = torch.where(s.LAIEXP < 6.0, s.LAIEXP * p.RGRLAI * DTEFF * factor, 0.0)
        r.GLASOL = torch.where(s.LAIEXP < 6.0, r.GRLV * r.SLAT, 0.0)

        r.SLAT = torch.where(
            (s.LAIEXP < 6.0) & (r.GRLV > 0.0),
            torch.min(r.GLAIEX, r.GLASOL) / r.GRLV.clamp(min=EPS),
            r.SLAT,
        )
        # Evaluate to 0 when _emerging
        r.GRLV = torch.where(_emerging, 0.0, r.GRLV)
        r.DSLV1 = torch.where(_emerging, 0.0, r.DSLV1)
        r.DSLV2 = torch.where(_emerging, 0.0, r.DSLV2)
        r.DSLV3 = torch.where(_emerging, 0.0, r.DSLV3)
        r.DSLV4 = torch.where(_emerging, 0.0, r.DSLV4)
        r.DSLV = torch.where(_emerging, 0.0, r.DSLV)
        r.DALV = torch.where(_emerging, 0.0, r.DALV)
        r.DRLV = torch.where(_emerging, 0.0, r.DRLV)
        r.SLAT = torch.where(_emerging, 0.0, r.SLAT)
        r.FYSAGE = torch.where(_emerging, 0.0, r.FYSAGE)
        r.GLAIEX = torch.where(_emerging, 0.0, r.GLAIEX)
        r.GLASOL = torch.where(_emerging, 0.0, r.GLASOL)

        self.rates._update_kiosk()

    def process_LV(
        self,
        tLV: torch.Tensor,
        tLVAGE: torch.Tensor,
        tSLA: torch.Tensor,
        tDRLV: torch.Tensor,
        tLVPOINTER: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process tLV, tLVAGE, tSLA tensors based on demand tDRLV (shape: batch_size,).
        tLV, tLVAGE, tSLA: tensors of shape (batch_size, history_length).
        Returns updated tensors and tDRLV.
        """
        batch_size, history_length = tLV.shape

        tLV_new = tLV.clone()
        tLVAGE_new = tLVAGE.clone()
        tSLA_new = tSLA.clone()
        tDRLV_new = tDRLV.clone()

        # Process from end to start (right to left)
        for i in reversed(range(history_length)):
            remaining = tDRLV_new > 0

            if not remaining.any():
                break
            LVweight = tLV_new[:, i]

            # Case 1: Full removal (demand >= LVweight)
            full_remove = (tDRLV_new >= LVweight) & remaining
            tDRLV_new = torch.where(full_remove, tDRLV_new - LVweight, tDRLV_new)  # Not sure about this line
            tLV_new[:, i] = torch.where(full_remove, torch.tensor(0.0, device=tLV.device), tLV_new[:, i])
            tLVAGE_new[:, i] = torch.where(full_remove, torch.tensor(0.0, device=tLVAGE.device), tLVAGE_new[:, i])
            tSLA_new[:, i] = torch.where(full_remove, torch.tensor(0.0, device=tSLA_new.device), tSLA_new[:, i])
            tLVPOINTER = torch.where(full_remove & (i < tLVPOINTER), tLVPOINTER - 1, tLVPOINTER)

            # Case 2: Partial removal (demand < LVweight)
            partial_remove = (tDRLV_new < LVweight) & remaining
            tLV_new[:, i] = torch.where(partial_remove, LVweight - tDRLV_new, tLV_new[:, i])
            tDRLV_new = torch.where(partial_remove, torch.tensor(0.0, device=tDRLV.device), tDRLV_new)

        return tLV_new, tLVAGE_new, tSLA_new, tLVPOINTER

    def integrate(self, day: date, delt: float = 1.0) -> None:
        """Integrate state rates to new state"""

        p = self.params
        r = self.rates
        s = self.states

        tLV = self.LV
        tSLA = self.SLA
        tLVAGE = self.LVAGE
        tDRLV = r.DRLV
        tLVPOINTER = self.LVPOINTER

        tLV, tLVAGE, tSLA, tLVPOINTER = self.process_LV(tLV, tLVAGE, tSLA, tDRLV, tLVPOINTER)

        updates = torch.zeros_like(tLVAGE).to(self.device)

        mask = torch.arange(tLV.shape[1], device=self.device).unsqueeze(0) < (tLVPOINTER.unsqueeze(1) + 1)
        updates[mask] = r.FYSAGE.unsqueeze(1).expand_as(updates)[mask]
        tLVAGE = tLVAGE + updates

        tLV = tensor_appendleft(tLV, r.GRLV)
        tSLA = tensor_appendleft(tSLA, r.SLAT)
        tLVAGE = tensor_appendleft(tLVAGE, torch.zeros((self.num_models,)).to(self.device))
        tLVPOINTER = tLVPOINTER + 1

        s.LASUM = torch.sum(tLV * tSLA, dim=1)
        s.LAI = self._calc_LAI()
        s.LAIMAX = torch.max(s.LAI, s.LAIMAX)
        s.LAIEXP = s.LAIEXP + r.GLAIEX

        s.WLV = torch.sum(tLV, dim=1)
        s.DWLV = s.DWLV + r.DRLV
        s.TWLV = s.WLV + s.DWLV

        self.LV = tLV
        self.SLA = tSLA
        self.LVAGE = tLVAGE
        self.LVPOINTER = tLVPOINTER

        self.states._update_kiosk()

    def reset(self, day: date, inds: torch.Tensor = None) -> None:
        """Reset states and rates"""
        inds = torch.ones((self.num_models,), device=self.device).to(torch.bool) if inds is None else inds

        p = self.params
        s = self.states
        r = self.rates
        k = self.kiosk

        WLV = (p.TDWI * (1 - k.FR)) * k.FL
        DWLV = 0.0
        TWLV = WLV + DWLV

        SLA = torch.zeros((self.num_models, self.MAX_LEN)).to(self.device)
        LVAGE = torch.zeros((self.num_models, self.MAX_LEN)).to(self.device)
        LV = torch.zeros((self.num_models, self.MAX_LEN)).to(self.device)
        SLA[:, 0] = p.SLATB(k.DVS)
        LV[:, 0] = WLV

        LAIEM = LV[:, 0] * SLA[:, 0]
        LASUM = LAIEM
        LAIEXP = LAIEM
        LAIMAX = LAIEM
        SAI = PAI = 0
        if "SAI" in self.kiosk:
            SAI = k.SAI
        if "PAI" in self.kiosk:
            PAI = k.PAI
        LAI = LASUM + SAI + PAI

        self.LV = torch.where(inds.unsqueeze(1), LV, self.LV).detach()
        self.SLA = torch.where(inds.unsqueeze(1), SLA, self.SLA).detach()
        self.LVAGE = torch.where(inds.unsqueeze(1), LVAGE, self.LVAGE).detach()
        self.LVPOINTER = torch.where(inds, 1.0, self.LVPOINTER).detach()

        s.LAIEM = torch.where(inds, LAIEM, s.LAIEM).detach()
        s.LASUM = torch.where(inds, LASUM, s.LASUM).detach()
        s.LAIEXP = torch.where(inds, LAIEXP, s.LAIEXP).detach()
        s.LAIMAX = torch.where(inds, LAIMAX, s.LAIMAX).detach()
        s.LAI = torch.where(inds, LAI, s.LAI).detach()
        s.WLV = torch.where(inds, WLV, s.WLV).detach()
        s.DWLV = torch.where(inds, DWLV, s.DWLV).detach()
        s.TWLV = torch.where(inds, TWLV, s.TWLV).detach()

        r.GRLV = torch.where(inds, 0.0, r.GRLV).detach()
        r.DSLV1 = torch.where(inds, 0.0, r.DSLV1).detach()
        r.DSLV2 = torch.where(inds, 0.0, r.DSLV2).detach()
        r.DSLV3 = torch.where(inds, 0.0, r.DSLV3).detach()
        r.DSLV4 = torch.where(inds, 0.0, r.DSLV4).detach()
        r.DSLV = torch.where(inds, 0.0, r.DSLV).detach()
        r.DALV = torch.where(inds, 0.0, r.DALV).detach()
        r.DRLV = torch.where(inds, 0.0, r.DRLV).detach()
        r.SLAT = torch.where(inds, 0.0, r.SLAT).detach()
        r.FYSAGE = torch.where(inds, 0.0, r.FYSAGE).detach()
        r.GLAIEX = torch.where(inds, 0.0, r.GLAIEX).detach()
        r.GLASOL = torch.where(inds, 0.0, r.GLASOL).detach()

        self.states._update_kiosk()
        self.rates._update_kiosk()

    def get_output(self, va: list[str] = None) -> torch.Tensor:
        """
        Return the output
        """
        if va is None:
            return self.states.LAI
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
        return {"LV": self.LV, "SLA": self.SLA, "LVAGE": self.LVAGE}

    def set_model_specific_params(self, k: str, v: torch.Tensor) -> None:
        """
        Set the specific parameters to handle overrides as needed
        Like casting to ints
        """
        setattr(self.params, k, v)
