"""All prior physical, summary, gate, full native trace and pose checks, including the finer level."""
from pathlib import Path
import boundary_runtime as rt
from midpoint_gates import admission,learning_admission

def check_row(folder,row,contract,native=False):
    from audit_fix import check_physics
    from midpoint_trace_audit import audit
    from pose_evidence import check_pose
    path=folder/row['summary'];assert rt.sha(path)==row['summary_sha256']
    s=rt.read(path);m=rt.metrics(s)
    assert m==row['metrics'] and s['max_pre_apex_mimic_v_rad_s']==row['pre_apex_v_rad_s']
    assert s['max_mimic_v_rad_s']==row['max_mimic_v_rad_s'] and s['sample_kind']==row['sample_kind']
    assert s['external_wrench']==row['external_wrench']
    check_physics(row['physics_receipt'],'rotor5ms')
    anchors=contract['anchors_native'] if native else contract['anchors_batch']
    assert admission(m,anchors,row['pre_apex_v_rad_s'])==row['strict_admission']
    assert learning_admission(s,anchors,row['pre_apex_v_rad_s'])==row['learning_admission']
    if native:
        path=folder/row['trace'];assert rt.sha(path)==row['trace_sha256']
        assert audit(path,s,row['mass_kg'],contract['profile'],row['before'],row['after'],'rotor5ms')==row['audits']
    if row['sample_kind']=='deterministic_evaluation':
        check_pose(folder/row['poses'],row['poses_sha256'],row['metrics']['worlds'])
