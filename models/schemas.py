

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator



class NodeType(str, Enum):
    SYMPTOM      = "symptom"
    RISK_FACTOR  = "risk_factor"
    HYPOTHESIS   = "hypothesis"
    FINDING      = "finding"         
    MISSING      = "missing"         
    BIAS_FLAG    = "bias_flag"


class BiasType(str, Enum):
    PREMATURE_CLOSURE = "premature_closure"
    ANCHORING_BIAS    = "anchoring_bias"


class TimerAction(str, Enum):
    START = "start"
    STOP  = "stop"



class AddNodeData(BaseModel):
    id:            str      = Field(..., description="ID unik node, snake_case. Contoh: sym_batuk_kronik")
    label:         str      = Field(..., description="Label yang tampil di node. Maks 40 karakter.")
    type:          NodeType = Field(..., description="Kategori node")
    checklist_ref: Optional[str] = Field(
        None,
        description="Referensi ID item checklist OSCE yang dipenuhi oleh node ini, misal 'anamnesis_1'"
    )

    @field_validator("id")
    @classmethod
    def id_must_be_snake_case(cls, v: str) -> str:
        import re
        if not re.match(r'^[a-z][a-z0-9_]*$', v):
            raise ValueError(
                f"Node id harus snake_case lowercase, dapat mengandung huruf, angka, underscore. Got: '{v}'"
            )
        return v

    @field_validator("label")
    @classmethod
    def label_max_length(cls, v: str) -> str:
        if len(v) > 50:
            raise ValueError(f"Label terlalu panjang ({len(v)} chars). Maks 50.")
        return v


class AddEdgeData(BaseModel):
    id:     str           = Field(..., description="ID unik edge. Konvensi: edge_{source}_{target}")
    source: str           = Field(..., description="ID node asal (harus sudah ada di canvas)")
    target: str           = Field(..., description="ID node tujuan (harus sudah ada di canvas)")
    label:  Optional[str] = Field(None, description="Label relasi, contoh: 'menyebabkan', 'faktor risiko'")


class TriggerHintData(BaseModel):
    checklist_id:    str           = Field(..., description="ID item checklist yang belum digali")
    hint_message:    str           = Field(..., description="Pertanyaan reflektif Sokrates. Jangan jawab langsung.")
    target_node_id:  Optional[str] = Field(None, description="ID node yang di-highlight/pulse di whiteboard")


class FlagBiasData(BaseModel):
    bias_type:            BiasType = Field(..., description="Tipe bias kognitif yang terdeteksi")
    description:          str      = Field(..., description="Penjelasan mengapa bias ini terdeteksi, kontekstual")
    affected_hypothesis:  Optional[str] = Field(
        None,
        description="Label hipotesis yang terdampak bias, untuk highlight di canvas"
    )


class UpdateHypothesisData(BaseModel):
    id:    str = Field(..., description="ID node hipotesis yang akan diupdate")
    label: str = Field(..., description="Label baru node hipotesis")



class AddNodeAction(BaseModel):
    type: Literal["add_node"]
    data: AddNodeData

class AddEdgeAction(BaseModel):
    type: Literal["add_edge"]
    data: AddEdgeData

class TriggerHintAction(BaseModel):
    type: Literal["trigger_hint"]
    data: TriggerHintData

class FlagBiasAction(BaseModel):
    type: Literal["flag_bias"]
    data: FlagBiasData

class UpdateHypothesisAction(BaseModel):
    type: Literal["update_hypothesis"]
    data: UpdateHypothesisData

class NoAction(BaseModel):
    type: Literal["no_action"]
    data: None = None


WhiteboardAction = Annotated[
    Union[
        AddNodeAction,
        AddEdgeAction,
        TriggerHintAction,
        FlagBiasAction,
        UpdateHypothesisAction,
        NoAction,
    ],
    Field(discriminator="type"),
]



class SimulationStep(BaseModel):

    text:         str                  = Field(
        ...,
        description=(
            "Dialog pasien virtual yang menjawab pertanyaan mahasiswa. "
            "Bahasa Indonesia awam, natural, tidak menyebut nama penyakit. "
            "Atau narasi sistem (misal: sesi dimulai, sesi berakhir)."
        )
    )
    actions:      list[WhiteboardAction] = Field(
        default_factory=list,
        description="Daftar perintah whiteboard. Boleh kosong. Urutan penting: dieksekusi berurutan."
    )
    timer_action: Optional[TimerAction] = Field(
        None,
        alias="timerAction",
        description="'start' saat sesi dimulai, 'stop' saat sesi berakhir atau waktu habis."
    )

    model_config = {
        "populate_by_name": True,   # allow both 'timer_action' and 'timerAction'
        "json_schema_extra": {
            "example": {
                "text": "Sudah sekitar 3 minggu dok, batuknya berdahak warna kuning kehijauan.",
                "actions": [
                    {
                        "type": "add_node",
                        "data": {
                            "id": "sym_batuk_kronik",
                            "label": "Batuk berdahak (3 minggu)",
                            "type": "symptom",
                            "checklist_ref": "anamnesis_1"
                        }
                    },
                    {
                        "type": "add_edge",
                        "data": {
                            "id": "edge_sym_batuk_kronik_hyp_tb",
                            "source": "sym_batuk_kronik",
                            "target": "hyp_tb_paru",
                            "label": "gejala kardinal"
                        }
                    }
                ],
                "timerAction": None
            }
        }
    }

    @model_validator(mode="after")
    def validate_action_limits(self) -> "SimulationStep":
        """
        Validasi:
        - Maks 1 flag_bias per respons
        - Maks 2 add_node per respons  
        - Edge hanya boleh ada jika node source & target valid (warning only)
        """
        bias_count = sum(1 for a in self.actions if a.type == "flag_bias")
        if bias_count > 1:
            raise ValueError(
                f"Maksimal 1 flag_bias per SimulationStep, ditemukan {bias_count}. "
                "Gabungkan bias atau prioritaskan yang paling kritikal."
            )

        node_count = sum(1 for a in self.actions if a.type == "add_node")
        if node_count > 3:
            raise ValueError(
                f"Maksimal 3 add_node per SimulationStep, ditemukan {node_count}. "
                "Pecah menjadi beberapa giliran agar animasi tidak terlalu cepat."
            )

        return self

    def to_frontend_dict(self) -> dict:
        """
        Serialize ke format yang siap dikirim ke frontend via WebSocket.
        Menggunakan alias 'timerAction' sesuai TypeScript interface.
        """
        return self.model_dump(by_alias=True, exclude_none=False)



class BiasAnalysis(BaseModel):
    has_premature_closure: bool = False
    has_anchoring:         bool = False
    closure_reason:        Optional[str] = None
    anchoring_reason:      Optional[str] = None
    anchored_hypothesis:   Optional[str] = None


class SessionContext(BaseModel):
    case_id:              str
    turn_number:          int
    completed_checklist:  list[str] = Field(default_factory=list)
    node_ids_on_canvas:   list[str] = Field(default_factory=list)
    detected_biases:      list[BiasType] = Field(default_factory=list)
    student_hypothesis:   Optional[str] = None
    elapsed_seconds:      int = 0

    @property
    def completion_rate(self) -> float:
        """Persentase checklist yang sudah selesai (0.0 - 1.0)."""
        # Total checklist RESP-001: 9 anamnesis + 4 PE + 3 workup = 16
        TOTAL = 16
        return len(self.completed_checklist) / TOTAL