"""
Class to calculate various nutrient relates stress factors:

Written by: Will Solow, 2025
"""

from datetime import date
import torch

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


class NPK_Stress_TensorBatch(BatchTensorModel):
    """Implementation of NPK stress calculation through [NPK]nutrition index."""

    class Parameters(ParamTemplate):
        NMAXLV_TB = TensorBatchAfgenTrait()
        PMAXLV_TB = TensorBatchAfgenTrait()
        KMAXLV_TB = TensorBatchAfgenTrait()
        NCRIT_FR = Tensor(-99.0)
        PCRIT_FR = Tensor(-99.0)
        KCRIT_FR = Tensor(-99.0)
        NMAXRT_FR = Tensor(-99.0)
        NMAXST_FR = Tensor(-99.0)
        PMAXST_FR = Tensor(-99.0)
        PMAXRT_FR = Tensor(-99.0)
        KMAXRT_FR = Tensor(-99.0)
        KMAXST_FR = Tensor(-99.0)
        NRESIDLV = Tensor(-99.0)
        NRESIDST = Tensor(-99.0)
        PRESIDLV = Tensor(-99.0)
        PRESIDST = Tensor(-99.0)
        KRESIDLV = Tensor(-99.0)
        KRESIDST = Tensor(-99.0)
        NLUE_NPK = Tensor(-99.0)

    class RateVariables(RatesTemplate):
        NNI = Tensor(-99.0)
        PNI = Tensor(-99.0)
        KNI = Tensor(-99.0)
        NPKI = Tensor(-99.0)
        RFNPK = Tensor(-99.0)

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

        self.rates = self.RateVariables(num_models=self.num_models, kiosk=self.kiosk, publish=["NPKI", "NNI"])

        self.min_tensor = torch.tensor([0.001]).to(self.device)
        self.max_tensor = torch.tensor([1.0]).to(self.device)

    def __call__(
        self, day: date, drv: DFTensorWeatherDataContainer, _emerging: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Callable to compute stress parameters"""
        p = self.params
        r = self.rates
        k = self.kiosk

        NMAXLV = p.NMAXLV_TB(k.DVS)
        PMAXLV = p.PMAXLV_TB(k.DVS)
        KMAXLV = p.KMAXLV_TB(k.DVS)

        NMAXST = p.NMAXST_FR * NMAXLV
        PMAXST = p.PMAXRT_FR * PMAXLV
        KMAXST = p.KMAXST_FR * KMAXLV

        VBM = k.WLV + k.WST

        NcriticalLV = p.NCRIT_FR * NMAXLV * k.WLV
        NcriticalST = p.NCRIT_FR * NMAXST * k.WST

        PcriticalLV = p.PCRIT_FR * PMAXLV * k.WLV
        PcriticalST = p.PCRIT_FR * PMAXST * k.WST

        KcriticalLV = p.KCRIT_FR * KMAXLV * k.WLV
        KcriticalST = p.KCRIT_FR * KMAXST * k.WST

        NcriticalVBM = torch.where(VBM > 0.0, (NcriticalLV + NcriticalST) / VBM.clamp(min=EPS), 0.0)
        PcriticalVBM = torch.where(VBM > 0.0, (PcriticalLV + PcriticalST) / VBM.clamp(min=EPS), 0.0)
        KcriticalVBM = torch.where(VBM > 0.0, (KcriticalLV + KcriticalST) / VBM.clamp(min=EPS), 0.0)

        NconcentrationVBM = torch.where(VBM > 0.0, (k.NAMOUNTLV + k.NAMOUNTST) / VBM.clamp(min=EPS), 0.0)
        PconcentrationVBM = torch.where(VBM > 0.0, (k.PAMOUNTLV + k.PAMOUNTST) / VBM.clamp(min=EPS), 0.0)
        KconcentrationVBM = torch.where(VBM > 0.0, (k.KAMOUNTLV + k.KAMOUNTST) / VBM.clamp(min=EPS), 0.0)

        NresidualVBM = torch.where(VBM > 0.0, (k.WLV * p.NRESIDLV + k.WST * p.NRESIDST) / VBM.clamp(min=EPS), 0.0)
        PresidualVBM = torch.where(VBM > 0.0, (k.WLV * p.PRESIDLV + k.WST * p.PRESIDST) / VBM.clamp(min=EPS), 0.0)
        KresidualVBM = torch.where(VBM > 0.0, (k.WLV * p.KRESIDLV + k.WST * p.KRESIDST) / VBM.clamp(min=EPS), 0.0)

        r.NNI = torch.where(
            (NcriticalVBM - NresidualVBM) > 0.0,
            torch.clamp(
                (NconcentrationVBM - NresidualVBM) / (NcriticalVBM - NresidualVBM).clamp(min=EPS),
                self.min_tensor,
                self.max_tensor,
            ),
            0.001,
        )
        r.PNI = torch.where(
            (PcriticalVBM - PresidualVBM) > 0.0,
            torch.clamp(
                (PconcentrationVBM - PresidualVBM) / (PcriticalVBM - PresidualVBM).clamp(min=EPS),
                self.min_tensor,
                self.max_tensor,
            ),
            0.001,
        )
        r.KNI = torch.where(
            (KcriticalVBM - KresidualVBM) > 0.0,
            torch.clamp(
                (KconcentrationVBM - KresidualVBM) / (KcriticalVBM - KresidualVBM).clamp(min=EPS),
                self.min_tensor,
                self.max_tensor,
            ),
            0.001,
        )

        r.NPKI = torch.min(torch.min(r.NNI, r.PNI), r.KNI)

        r.RFNPK = torch.clamp(
            1.0 - (p.NLUE_NPK * (1.0001 - r.NPKI) ** 2),
            torch.tensor([0.0]).to(self.device),
            self.max_tensor,
        )

        r.NNI = torch.where(_emerging, 0.0, r.NNI)
        r.PNI = torch.where(_emerging, 0.0, r.PNI)
        r.KNI = torch.where(_emerging, 0.0, r.KNI)
        r.NPKI = torch.where(_emerging, 0.0, r.NPKI)
        r.RFNPK = torch.where(_emerging, 0.0, r.RFNPK)

        self.rates._update_kiosk()

        return r.NNI, r.NPKI, r.RFNPK

    def reset(self, day: date, inds: torch.Tensor = None) -> None:
        """Reset states and rates"""
        inds = torch.ones((self.num_models,), device=self.device).to(torch.bool) if inds is None else inds
        r = self.rates

        r.NNI = torch.where(inds, 0.0, r.NNI).detach()
        r.PNI = torch.where(inds, 0.0, r.PNI).detach()
        r.KNI = torch.where(inds, 0.0, r.KNI).detach()
        r.NPKI = torch.where(inds, 0.0, r.NPKI).detach()
        r.RFNPK = torch.where(inds, 0.0, r.RFNPK).detach()

        r._update_kiosk()

    def get_output(self, va: list[str] = None) -> torch.Tensor:
        """
        Return the output
        """
        if va is None:
            return self.rates.NPKI
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
