"""Independently reconstruct admissions, selected history, traces and exits."""
import argparse
import zero_runtime as rt
from transfer_contract import BEFORE,AFTER,VARIANT,admission,promote,final_qualified


def check_row(folder,row,contract,native=False):
    from audit_fix import check_physics
    from zero_trace_audit import audit
    path=folder/row['summary']
    assert rt.sha(path)==row['summary_sha256']
    s=rt.read(path); m=rt.metrics(s)
    assert m==row['metrics'] and s['max_pre_apex_mimic_v_rad_s']==row['pre_apex_v_rad_s']
    assert s['max_mimic_v_rad_s']==row['max_mimic_v_rad_s']
    assert s['sample_kind']==row['sample_kind']
    assert s['external_wrench']==row['external_wrench']
    check_physics(row['physics_receipt'],VARIANT)
    anchors=contract['anchors_native'] if native else contract['anchors_batch']
    for key,learning in [('strict_admission',False),('learning_admission',True)]:
        assert admission(m,anchors,row['pre_apex_v_rad_s'],learning)==row[key]
    if native:
        path=folder/row['trace']; assert rt.sha(path)==row['trace_sha256']
        assert audit(path,s,row['mass_kg'],contract['profile'],row['before'],row['after'],VARIANT)==row['audits']
