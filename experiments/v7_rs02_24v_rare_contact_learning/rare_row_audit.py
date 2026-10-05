"""Preserve original physical/strict audits, recompute the new learning decision."""
from copy import deepcopy
import rare_runtime as rt
from rare_contract import admission,learning_admission
from repair_row_audit import check_row as inherited

def check_row(folder,row,contract,native=False):
    s=rt.read(folder/row['summary']);anchors=contract['anchors_native'] if native else contract['anchors_batch']
    original=deepcopy(row)
    original['learning_admission']=admission(row['metrics'],anchors,row['pre_apex_v_rad_s'],True)
    inherited(folder,original,contract,native)
    assert learning_admission(s,anchors,row['pre_apex_v_rad_s'])==row['learning_admission']
