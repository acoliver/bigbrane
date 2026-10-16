"""OAI-PMH page parsing: a real arXiv response trimmed to three records, plus synthetic edge cases."""
import pytest

from hybrid.literature.oai import OaiError, parse_list_records
from hybrid.tests.literature.oai_support import oai_error, oai_page, record
from hybrid.tests.literature.support import fixture_text


def test_real_response_parses_records():
    page = parse_list_records(fixture_text("oai_list_records.xml"))

    assert page.response_date == "2026-10-11T00:37:04Z"
    assert [r.arxiv_id for r in page.records] == ["1808.06723", "2303.06704", "2504.06209"]
    assert page.deleted_ids == () and page.malformed_ids == ()
    assert page.resumption_token is None

    first = page.records[0]
    assert first.title == "Modes of Information Flow"
    assert first.authors == ("Ryan G. James", "Blanca Daniella Mansante Ayala", "Bahti Zakirov", "James P. Crutchfield")
    assert first.categories == ("cond-mat.stat-mech", "cs.CR", "cs.IT", "math.IT", "nlin.CD")
    assert first.created == "2026-09-29" and first.updated == "2026-10-01" and first.datestamp == "2026-10-01"
    assert first.abstract.startswith("Information flow between components of a system takes many forms")
    assert "Y's past" in first.abstract and "\n" not in first.abstract


def test_resumption_token_and_list_size():
    page = parse_list_records(oai_page([record("2401.00001", "A")], token="abc|123", list_size=5000))
    assert page.resumption_token == "abc|123"
    assert page.complete_list_size == 5000


def test_empty_resumption_token_means_last_page():
    assert parse_list_records(oai_page([record("2401.00001", "A")])).resumption_token is None


def test_deleted_records_are_reported_by_id():
    page = parse_list_records(oai_page([], deleted=("2401.00009", "hep-th/9901001")))
    assert page.records == ()
    assert page.deleted_ids == ("2401.00009", "hep-th/9901001")


def test_record_without_title_is_malformed_not_fatal():
    page = parse_list_records(oai_page([record("2401.00001", "Fine"), record("2401.00002", "")]))
    assert [r.arxiv_id for r in page.records] == ["2401.00001"]
    assert page.malformed_ids == ("oai:arXiv.org:2401.00002",)


def test_error_element_raises_with_code():
    with pytest.raises(OaiError) as caught:
        parse_list_records(oai_error("badResumptionToken", "expired"))
    assert caught.value.code == "badResumptionToken"


def test_malformed_xml_raises_value_error():
    with pytest.raises(ValueError, match="malformed OAI-PMH"):
        parse_list_records("<OAI-PMH><oops></OAI-PMH>")
