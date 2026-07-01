import json
import sys
from pathlib import Path

import pytest

# Path setup
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.schemas import (
    AddNodeAction,
    AddNodeData,
    AddEdgeAction,
    AddEdgeData,
    FlagBiasAction,
    FlagBiasData,
    TriggerHintAction,
    TriggerHintData,
    NoAction,
    SimulationStep,
    NodeType,
    BiasType,
    TimerAction,
    BiasAnalysis,
    SessionContext,
)
from utils.response_parser import (
    parse_llm_response,
    parse_and_inject_bias,
    validate_step,
    _clean_raw,
    _extract_json,
    _normalize_fields,
    _to_snake_case,
)


# ══════════════════════════════════════════════════════════════════════
# SCHEMAS TESTS
# ══════════════════════════════════════════════════════════════════════

class TestNodeType:
    def test_valid_enum_values(self):
        assert NodeType.SYMPTOM.value     == "symptom"
        assert NodeType.RISK_FACTOR.value == "risk_factor"
        assert NodeType.HYPOTHESIS.value  == "hypothesis"
        assert NodeType.FINDING.value     == "finding"
        assert NodeType.BIAS_FLAG.value   == "bias_flag"

    def test_enum_from_string(self):
        assert NodeType("symptom")     == NodeType.SYMPTOM
        assert NodeType("risk_factor") == NodeType.RISK_FACTOR


class TestAddNodeData:
    def test_valid_node(self):
        node = AddNodeData(id="sym_batuk_kronik", label="Batuk berdahak (3 minggu)", type=NodeType.SYMPTOM)
        assert node.id == "sym_batuk_kronik"
        assert node.type == NodeType.SYMPTOM

    def test_invalid_id_camelcase(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            AddNodeData(id="symBatukKronik", label="Test", type=NodeType.SYMPTOM)

    def test_invalid_id_spaces(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            AddNodeData(id="sym batuk", label="Test", type=NodeType.SYMPTOM)

    def test_label_too_long(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            AddNodeData(id="sym_x", label="A" * 51, type=NodeType.SYMPTOM)

    def test_optional_checklist_ref(self):
        node = AddNodeData(id="sym_x", label="Test", type=NodeType.SYMPTOM)
        assert node.checklist_ref is None


class TestSimulationStep:
    def test_minimal_valid_step(self):
        step = SimulationStep(
            text="Sudah sekitar 3 minggu Dok.",
            actions=[],
        )
        assert step.text == "Sudah sekitar 3 minggu Dok."
        assert step.actions == []
        assert step.timer_action is None

    def test_step_with_add_node(self):
        step = SimulationStep(
            text="Saya batuk berdahak Dok.",
            actions=[
                AddNodeAction(
                    type="add_node",
                    data=AddNodeData(id="sym_batuk", label="Batuk berdahak", type=NodeType.SYMPTOM)
                )
            ]
        )
        assert len(step.actions) == 1
        assert step.actions[0].type == "add_node"

    def test_too_many_bias_flags_raises(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="Maksimal 1 flag_bias"):
            SimulationStep(
                text="Test",
                actions=[
                    FlagBiasAction(
                        type="flag_bias",
                        data=FlagBiasData(bias_type=BiasType.PREMATURE_CLOSURE, description="Test 1")
                    ),
                    FlagBiasAction(
                        type="flag_bias",
                        data=FlagBiasData(bias_type=BiasType.ANCHORING_BIAS, description="Test 2")
                    ),
                ]
            )

    def test_too_many_add_nodes_raises(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="Maksimal 3 add_node"):
            SimulationStep(
                text="Test",
                actions=[
                    AddNodeAction(
                        type="add_node",
                        data=AddNodeData(id=f"sym_{i}", label=f"Node {i}", type=NodeType.SYMPTOM)
                    )
                    for i in range(4)
                ]
            )

    def test_to_frontend_dict_uses_alias(self):
        step = SimulationStep(text="Test", actions=[], timerAction=TimerAction.START)
        d = step.to_frontend_dict()
        assert "timerAction" in d
        assert d["timerAction"] == "start"

    def test_timer_action_alias_both_ways(self):
        # Bisa dibuat dengan alias
        step1 = SimulationStep(text="Test", actions=[], timerAction="start")
        step2 = SimulationStep(text="Test", actions=[], timer_action="start")
        assert step1.timer_action == TimerAction.START
        assert step2.timer_action == TimerAction.START


class TestSessionContext:
    def test_completion_rate_empty(self):
        ctx = SessionContext(case_id="RESP-001", turn_number=1)
        assert ctx.completion_rate == 0.0

    def test_completion_rate_partial(self):
        ctx = SessionContext(
            case_id="RESP-001",
            turn_number=5,
            completed_checklist=["a1", "a2", "a3", "a4"]   # 4 out of 16
        )
        assert ctx.completion_rate == pytest.approx(0.25)


# ══════════════════════════════════════════════════════════════════════
# RESPONSE PARSER TESTS
# ══════════════════════════════════════════════════════════════════════

class TestCleanRaw:
    def test_strips_json_fence(self):
        raw = '```json\n{"text": "test"}\n```'
        assert _clean_raw(raw) == '{"text": "test"}'

    def test_strips_plain_fence(self):
        raw = '```\n{"text": "test"}\n```'
        assert _clean_raw(raw) == '{"text": "test"}'

    def test_strips_prefix_text(self):
        raw = 'Here is the JSON:\n{"text": "test"}'
        result = _clean_raw(raw)
        assert result.startswith("{")

    def test_strips_suffix_text(self):
        raw = '{"text": "test"}\nSome trailing text.'
        result = _clean_raw(raw)
        assert result.endswith("}")

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _clean_raw("")

    def test_no_brace_raises(self):
        with pytest.raises(ValueError, match="JSON object"):
            _clean_raw("just some text without braces")


class TestExtractJson:
    def test_valid_json(self):
        result = _extract_json('{"text": "hello", "actions": []}')
        assert result["text"] == "hello"

    def test_trailing_comma_repaired(self):
        raw = '{"text": "hello", "actions": [],}'
        result = _extract_json(raw)
        assert result["text"] == "hello"

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            _extract_json("not json at all {{{{")


class TestNormalizeFields:
    def test_patient_dialogue_alias(self):
        data = {"patient_dialogue": "Hello Dok", "actions": []}
        result = _normalize_fields(data)
        assert "text" in result
        assert result["text"] == "Hello Dok"

    def test_whiteboard_actions_alias(self):
        data = {"text": "Hello", "whiteboard_actions": []}
        result = _normalize_fields(data)
        assert "actions" in result

    def test_missing_actions_defaults_to_empty_list(self):
        data = {"text": "Hello"}
        result = _normalize_fields(data)
        assert result["actions"] == []

    def test_timer_action_normalized(self):
        data = {"text": "Hello", "actions": [], "timer_action": "START"}
        result = _normalize_fields(data)
        assert result["timerAction"] == "start"


class TestToSnakeCase:
    def test_basic(self):
        assert _to_snake_case("sym Batuk Kronik") == "sym_batuk_kronik"

    def test_camelcase(self):
        # CamelCase tidak secara otomatis dikonversi, tapi karakter invalid dihapus
        result = _to_snake_case("symBatuk")
        assert " " not in result

    def test_special_chars_removed(self):
        assert _to_snake_case("sym!@#batuk") == "symbatuk"

    def test_multiple_underscores_collapsed(self):
        assert _to_snake_case("sym___batuk") == "sym_batuk"


class TestParseLlmResponse:
    def _make_raw(self, **kwargs) -> str:
        """Helper: buat valid JSON string."""
        default = {
            "text": "Sudah 3 minggu Dok.",
            "actions": [],
            "timerAction": None,
        }
        default.update(kwargs)
        return json.dumps(default)

    def test_parse_minimal_valid(self):
        raw = self._make_raw()
        step = parse_llm_response(raw)
        assert step.text == "Sudah 3 minggu Dok."
        assert step.actions == []

    def test_parse_with_add_node(self):
        raw = self._make_raw(actions=[{
            "type": "add_node",
            "data": {"id": "sym_batuk", "label": "Batuk berdahak", "type": "symptom"}
        }])
        step = parse_llm_response(raw)
        assert len(step.actions) == 1
        assert step.actions[0].type == "add_node"
        assert step.actions[0].data.id == "sym_batuk"

    def test_parse_with_markdown_fence(self):
        raw = '```json\n' + self._make_raw() + '\n```'
        step = parse_llm_response(raw)
        assert isinstance(step, SimulationStep)

    def test_parse_node_type_alias(self):
        """'vital_sign' harus dinormalisasi ke 'finding'."""
        raw = self._make_raw(actions=[{
            "type": "add_node",
            "data": {"id": "fn_td", "label": "TD 110/70", "type": "vital_sign"}
        }])
        step = parse_llm_response(raw)
        assert step.actions[0].data.type == NodeType.FINDING

    def test_parse_action_type_alias(self):
        """'addNode' (camelCase dari LLM) harus dinormalisasi ke 'add_node'."""
        raw = self._make_raw(actions=[{
            "type": "addNode",
            "data": {"id": "sym_x", "label": "Test", "type": "symptom"}
        }])
        step = parse_llm_response(raw)
        assert step.actions[0].type == "add_node"

    def test_parse_empty_string_returns_fallback(self):
        step = parse_llm_response("")
        assert isinstance(step, SimulationStep)
        assert "gangguan" in step.text.lower() or "kendala" in step.text.lower()

    def test_parse_completely_invalid_returns_fallback(self):
        step = parse_llm_response("This is definitely not JSON {{{{")
        assert isinstance(step, SimulationStep)

    def test_parse_patient_dialogue_alias(self):
        """LLM pakai 'patient_dialogue' instead of 'text'."""
        raw = json.dumps({
            "patient_dialogue": "Iya Dok sudah lama",
            "actions": []
        })
        step = parse_llm_response(raw)
        assert step.text == "Iya Dok sudah lama"

    def test_parse_edge_auto_id(self):
        """Edge tanpa 'id' harus auto-generate ID."""
        raw = self._make_raw(actions=[
            {"type": "add_node", "data": {"id": "sym_a", "label": "A", "type": "symptom"}},
            {"type": "add_node", "data": {"id": "hyp_b", "label": "B", "type": "hypothesis"}},
            {"type": "add_edge", "data": {"source": "sym_a", "target": "hyp_b", "label": "rel"}},
        ])
        step = parse_llm_response(raw)
        edge_action = next(a for a in step.actions if a.type == "add_edge")
        assert edge_action.data.id == "edge_sym_a_hyp_b"


class TestParseAndInjectBias:
    def test_inject_premature_closure(self):
        raw = json.dumps({
            "text": "Batuk sudah lama Dok.",
            "actions": [{"type": "no_action", "data": None}]
        })
        step = parse_and_inject_bias(
            raw            = raw,
            bias_type      = "premature_closure",
            bias_description = "Checklist baru 20% terisi.",
        )
        assert step.actions[0].type == "flag_bias"
        assert step.actions[0].data.bias_type == BiasType.PREMATURE_CLOSURE

    def test_inject_replaces_existing_bias(self):
        """Bias dari director menggantikan bias dari LLM."""
        raw = json.dumps({
            "text": "Test",
            "actions": [{
                "type": "flag_bias",
                "data": {
                    "bias_type": "anchoring_bias",
                    "description": "Dari LLM (harus diganti)"
                }
            }]
        })
        step = parse_and_inject_bias(
            raw              = raw,
            bias_type        = "premature_closure",
            bias_description = "Dari director (harus menang)",
        )
        bias_actions = [a for a in step.actions if a.type == "flag_bias"]
        assert len(bias_actions) == 1
        assert bias_actions[0].data.bias_type == BiasType.PREMATURE_CLOSURE

    def test_inject_bias_at_front(self):
        raw = json.dumps({
            "text": "Test",
            "actions": [
                {"type": "add_node", "data": {"id": "sym_x", "label": "X", "type": "symptom"}}
            ]
        })
        step = parse_and_inject_bias(
            raw              = raw,
            bias_type        = "anchoring_bias",
            bias_description = "Test anchoring",
            affected_hypothesis = "TB Paru",
        )
        assert step.actions[0].type == "flag_bias"
        assert step.actions[1].type == "add_node"


class TestValidateStep:
    def test_valid_step(self):
        step = SimulationStep(
            text="Test",
            actions=[
                AddNodeAction(
                    type="add_node",
                    data=AddNodeData(id="sym_a", label="A", type=NodeType.SYMPTOM)
                ),
                AddEdgeAction(
                    type="add_edge",
                    data=AddEdgeData(id="edge_sym_a_hyp_b", source="sym_a", target="hyp_b")
                )
            ]
        )
        # hyp_b tidak ada di step ini → warning
        is_valid, warnings = validate_step(step)
        assert not is_valid
        assert any("hyp_b" in w for w in warnings)

    def test_empty_text_warning(self):
        step = SimulationStep(text="   ", actions=[])
        is_valid, warnings = validate_step(step)
        assert not is_valid
        assert any("kosong" in w for w in warnings)


# ══════════════════════════════════════════════════════════════════════
# DIRECTOR BIAS DETECTION TESTS (tanpa API call)
# ══════════════════════════════════════════════════════════════════════

class TestDirectorBiasDetection:
    """
    Test bias detection logic di director tanpa memanggil OpenAI API.
    Hanya test metode _analyze_bias dan _extract_hypothesis_mention.
    """

    @pytest.fixture
    def mock_case_data(self):
        """Minimal case data untuk testing."""
        return {
            "case_id": "RESP-001",
            "title": "Test Case",
            "patient_presentation": {
                "name_placeholder": "Tn. A",
                "age": 45,
                "gender": "Laki-laki",
                "chief_complaint": "Batuk",
            },
            "illness_script": {
                "primary_diagnosis": "Tuberkulosis Paru",
                "enabling_conditions": [],
                "fault_pathophysiology": "",
                "consequences": {},
            },
            "differential_diagnoses": [
                {"diagnosis": "Pneumonia Komunitas", "distinguishing_features": ""},
                {"diagnosis": "Bronkitis Kronik", "distinguishing_features": ""},
                {"diagnosis": "Karsinoma Paru", "distinguishing_features": ""},
            ],
            "osce_checklist": {
                "anamnesis_items": [],
                "physical_exam_items": [],
                "expected_workup": [],
            },
            "common_bias_triggers": {
                "premature_closure_risk": "Tinggi",
                "anchoring_risk": "Tinggi",
            }
        }

    @pytest.fixture
    def director(self, mock_case_data, monkeypatch, tmp_path):
        """Director dengan mock case bank, tanpa OpenAI key."""
        # Buat case_bank.json sementara
        case_bank_path = tmp_path / "case_bank.json"
        case_bank_path.write_text(
            json.dumps([mock_case_data]), encoding="utf-8"
        )

        # Patch path dan env
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
        monkeypatch.setattr(
            "agent.director.CASE_BANK_PATH",
            case_bank_path
        )

        from agent.director import SimulationDirector
        d = SimulationDirector.__new__(SimulationDirector)
        d.case_id    = "RESP-001"
        d.session_id = "test-session"
        d._case_data = mock_case_data
        d._context   = SessionContext(case_id="RESP-001", turn_number=0)
        d._history   = []
        d._session_start = __import__("time").time()
        d._last_hypothesis_mentioned = None
        d._consec_same_hyp_count = 0
        d._biases_this_session = []
        return d

    def test_premature_closure_detected(self, director):
        director._context.turn_number = 2
        director._context.completed_checklist = ["a1"]  # < 30% dari 16

        bias = director._analyze_bias("Saya yakin ini TB, langsung berikan obat OAT")
        assert bias.has_premature_closure is True
        assert "30%" in bias.closure_reason or "18%" in bias.closure_reason or "6%" in bias.closure_reason

    def test_no_premature_closure_with_sufficient_checklist(self, director):
        director._context.turn_number = 3
        # 6 items = 37.5% > 30% threshold
        director._context.completed_checklist = [f"a{i}" for i in range(6)]

        bias = director._analyze_bias("Diagnosis saya adalah TB Paru")
        assert bias.has_premature_closure is False

    def test_no_premature_closure_late_turn(self, director):
        director._context.turn_number = 8  # > PREMATURE_CLOSURE_MAX_TURN (6)
        director._context.completed_checklist = []

        bias = director._analyze_bias("Diagnosis adalah TB")
        assert bias.has_premature_closure is False

    def test_anchoring_bias_after_3_mentions(self, director):
        director._context.turn_number = 5
        director._last_hypothesis_mentioned = "Tuberkulosis Paru"
        director._consec_same_hyp_count = 2  # sudah 2x, akan jadi 3

        bias = director._analyze_bias("Ini pasti TB Paru, saya yakin")
        assert bias.has_anchoring is True
        assert "Tuberkulosis Paru" in bias.anchored_hypothesis

    def test_anchoring_not_flagged_twice(self, director):
        director._context.turn_number = 5
        director._last_hypothesis_mentioned = "Tuberkulosis Paru"
        director._consec_same_hyp_count = 2
        director._biases_this_session = [BiasType.ANCHORING_BIAS]  # sudah pernah flag

        bias = director._analyze_bias("Ini pasti TB Paru")
        assert bias.has_anchoring is False

    def test_hypothesis_extraction_tb(self, director):
        hyp = director._extract_hypothesis_mention("saya curiga ini tb paru")
        assert hyp is not None
        assert "TB" in hyp.upper() or "Tuberkulosis" in hyp

    def test_hypothesis_extraction_pneumonia(self, director):
        hyp = director._extract_hypothesis_mention("kemungkinan pneumonia komunitas")
        assert hyp is not None

    def test_hypothesis_extraction_none(self, director):
        hyp = director._extract_hypothesis_mention("batuk sudah berapa lama pak?")
        assert hyp is None