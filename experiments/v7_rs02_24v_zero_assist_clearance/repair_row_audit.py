"""Preserve all prior row checks and bind diagnostic poses to each evaluation."""
from zero_row_audit import check_row as inherited
from pose_evidence import check_pose

def check_row(folder,row,contract,native=False):
    inherited(folder,row,contract,native)
    if row['sample_kind']=='deterministic_evaluation':
        check_pose(folder/row['poses'],row['poses_sha256'],row['metrics']['worlds'])
