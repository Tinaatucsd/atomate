# coding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals
from fireworks.core.firework import Workflow
from atomate.vasp.fireworks.core import OptimizeFW, TransmuterFW, StaticFW
from atomate.vasp.powerups import add_additional_fields_to_taskdocs, add_tags
from pymatgen.io.vasp.sets import MVLGBSet, MVLSlabSet

__author__ = 'Hui Zheng'
__email__ = 'huz071@eng.ucsd.edu'


def get_gb_fw(bulk_structure, gb_gen_params, db_file=None,
              vasp_input_set=None, parents=None, vasp_cmd="vasp", name=None):
    """
    The firework will generate a grain boundary from given bulk structure and parameters needed
    to generate a grain boundary (gb_gen_params), then return a TransmuterFW which include the
    relaxation for generated GB structure.

    Args:
        bulk_structure (Structure): Bulk structure could be primitive cell or conventional unit cell.
        gb_gen_params (dict): dictionary of gb generation parameters, used to generate gb from
            GBGenerator (which could be called by: from pymatgen.analysis.gb.grain import GrainBoundaryGenerator,
            where there are details of description for each parameter)
            gb_gen_params is necessary to get the gb that corresponds to the bulk structure.
            One example is given below:
              gb_s5_001_params = {"rotation_axis": [1, 0, 0], "rotation_angle": 36.86989764584402,
                        "expand_times": 2, "vacuum_thickness": 0.0, "normal": True,
                        "ratio": None, "plane": [1, 0, 0]}
        db_file (str): path to database file
        vasp_input_set (VaspInputSet): vasp_input_set corresponding to the grain boundary calculation
        parents (Fireworks or list of ints): parent FWs
        vasp_cmd (str): vasp command
        name (str): Name for the Firework

    Return:
        Firework
    """
    user_incar_settings = {"PREC": "Normal", "NPAR": 4, "ISMEAR": 1, "ENCUT": 400,
                           "ICHARG": 2, "LREAL": "False"}

    vis_for_transed_gb = vasp_input_set or MVLGBSet(bulk_structure, k_product=30,
                                                    user_incar_settings=user_incar_settings)
    transformations = ["GrainBoundaryTransformation"]
    trans_params = [gb_gen_params]

    name = bulk_structure.composition.reduced_formula
    if gb_gen_params.get('plane'):
        name += "_gb_plane{}".format(gb_gen_params['plane'])

    return TransmuterFW(structure=bulk_structure, transformations=transformations, name=name + "_transmuter",
                        transformation_params=trans_params, copy_vasp_outputs=True, db_file=db_file,
                        vasp_cmd=vasp_cmd, parents=parents, vasp_input_set=vis_for_transed_gb)


def get_wf_gb_from_bulk(bulk_structure, gb_gen_params, vasp_input_set=None, tag=None,
                        additional_info=None, db_file=None, vasp_cmd="vasp", name=None):
    """
    This is a workflow for grain boundary (gb) generation and calculation. Input bulk_structure and
     grain boundary (GB) generation parameters need to be specified, the workflow will relax the bulk structure
     first, then generate GB based on given parameters using TransmuterFW, which include the GB relaxation.



    Args:
        bulk_structure (Structure): bulk structure from which generate gbs after relaxation.
        gb_gen_params (dict): dictionary of gb generation parameters, used to generate gb. The
            details of description could be found in pymatgen.analysis.gb.gb import GBGenerator,
        tag (list): list of strings to tag the workflow, which will be inserted into the database
            make it easier to query data later. e.g.tag = ["mp-129"] to represent bcc-Mo wf.
        additional_info (Dict): the additional info of gb structure, which you want to add.
            The additional_info will be inserted into database.
        db_file (str): path to database file.
        vasp_cmd (str): vasp command

    Return:
        Workflow
    """

    additional_info = {} if not additional_info else additional_info
    fws, parents = [], []
    user_incar_settings = {"PREC": "Normal", "NPAR": 4, "ISMEAR": 1, "ENCUT": 400,
                           "ICHARG": 2, "LREAL": "False"}

    vis = vasp_input_set or MVLGBSet(bulk_structure, k_product=30, user_incar_settings=user_incar_settings)
    fws.append(OptimizeFW(structure=bulk_structure, vasp_input_set=vis,
                          vasp_cmd=vasp_cmd, db_file=db_file, name="bulk relaxation"))

    parents = fws[0]

    name = name or bulk_structure.composition.reduced_formula

    fws.append(get_gb_fw(bulk_structure=bulk_structure, gb_gen_params=gb_gen_params,
                         vasp_input_set=vis, db_file=db_file, vasp_cmd=vasp_cmd,
                         parents=parents, name=name))
    parents_2 = fws[-1]

    if gb_gen_params.get('plane'):
        name += "_gb_plane{}".format(gb_gen_params['plane'])

    static_fw = StaticFW(name=name + "_gb_static", vasp_cmd=vasp_cmd,
                         prev_calc_loc=True, db_file=db_file, parents=parents_2)
    fws.append(static_fw)

    wf = Workflow(fws, name="{} gb workflow, e.g., {}".format(len(fws), fws[0].name))
    wf_additional_info = add_additional_fields_to_taskdocs(original_wf=wf, update_dict=additional_info)
    wf_with_tag_info = add_tags(wf_additional_info, tags_list=tag)
    return wf_with_tag_info


def get_wf_gb(gb, vasp_input_set=None, vasp_input_set_params=None, tag=None,
              additional_info=None, db_file=None, vasp_cmd="vasp"):
    """
     The workflow will directly run the relaxation for the given GB structure, while the related
     additional info could be added into the database as user specified.

     Args:
        gb (Structure/GrainBoundary): Grain boundary structure, could be Structure, or GrainBoundary
         (GrainBoundary object can be imported from pymatgen.analysis.gb.grain)
        vasp_input_set (VaspInputSet): vasp_input_set corresponding to the grain boundary calculation
        tag (list): list of strings to tag the workflow, which will be inserted into the database
            make it easier to query data later. e.g.tag = ["mp-129"] to represent bcc-Mo wf.:
        additional_info (Dict): the additional info of gb structure, which you want to add.
            The additional_info will be inserted into database.
        db_file (str): path to database file.
        vasp_cmd (str): vasp command

    Return:
        Workflow
    """
    fws, parents = [], []

    user_incar_settings = {"PREC": "Normal", "NPAR": 4, "ISMEAR": 1, "ENCUT": 400, "ICHARG": 2,
                           "LVTOT": False}
    vis_for_given_gb = vasp_input_set or MVLSlabSet(gb, k_product=30, bulk=False, set_mix=False,
                                                    user_incar_settings=user_incar_settings)

    name = gb.composition.reduced_formula
    if getattr(gb, "gb_plane", None) and getattr(gb, "sigma", None):
        name += "_s{}_plane{}".format(gb.sigma, gb.gb_plane)
    fw = OptimizeFW(structure=gb, vasp_input_set=vis_for_given_gb, vasp_cmd=vasp_cmd,
                    job_type="double_relaxation_run", half_kpts_first_relax=True,
                    parents=parents, db_file=db_file, name=name + "_gb optimization")

    fws.append(fw)

    parents = fws[0]
    static_incar_settings = vasp_input_set_params or {"EDIFF": 0.0001, "AMIN": 0.01,
                                                      "ISPIN": 2, "ENCUT": 400, "NSW": 0, "EDIFFG": -0.02, "ICHARG": 2,
                                                      "NPAR": 4, "ALGO": "Normal", "IBRION": 1, "ISMEAR": 1,
                                                      "SIGMA": 0.02, "PREC": "Accurate", "NELM": 60, "LVTOT": False}

    # vis_for_static = MVLSlabSet(gb, k_product=45, bulk=False, set_mix=False,
    #                             user_incar_settings=static_incar_settings)

    static_fw = StaticFW(name=name + "_gb static", vasp_input_set_params=static_incar_settings,
                         vasp_cmd=vasp_cmd, prev_calc_loc=True, db_file=db_file, parents=parents)
    fws.append(static_fw)

    wf = Workflow(fws, name="{} gb workflow, e.g., {}".format(len(fws), fws[0].name))
    wf_additional_info = add_additional_fields_to_taskdocs(original_wf=wf, update_dict=additional_info)
    wf_with_tag_info = add_tags(wf_additional_info, tags_list=tag)
    return wf_with_tag_info


def gb_static_wf(gb, vasp_input_set=None, vasp_input_set_params=None, prev_calc_dir=None, tag=None,
                 additional_info=None, db_file=None, vasp_cmd="vasp"):
    """
     The workflow will directly run the relaxation for the given GB structure, while the related
     additional info could be added into the database as user specified.

     Args:
        gb (Structure/Gb): Grain boundary structure, could be Structure, or Gb (Gb object can
            be imported from pymatgen.analysis.gb.gb)
        vasp_input_set (VaspInputSet): vasp_input_set corresponding to the grain boundary calculation
        tag (list): list of strings to tag the workflow, which will be inserted into the database
            make it easier to query data later. e.g.tag = ["mp-129"] to represent bcc-Mo wf.:
        additional_info (Dict): the additional info of gb structure, which you want to add.
            The additional_info will be inserted into database.
        db_file (str): path to database file.
        vasp_cmd (str): vasp command

    Return:
        Workflow
    """
    fws = []
    static_incar_settings = vasp_input_set_params or {"EDIFF": 0.0001, "AMIN": 0.01,
                                                      "ISPIN": 2, "ENCUT": 400, "NSW": 0, "EDIFFG": -0.02, "ICHARG": 2,
                                                      "NPAR": 4, "ALGO": "Normal", "IBRION": 1, "ISMEAR": 1,
                                                      "SIGMA": 0.02, "PREC": "Accurate", "NELM": 60, "LVTOT": False}

    vis_for_static = vasp_input_set or MVLSlabSet(gb, k_product=45, bulk=False, set_mix=False,
                                                  user_incar_settings=static_incar_settings)
    name = gb.composition.reduced_formula
    if additional_info.get('gb_object'):
        name += "_gb_plane{}".format(additional_info['gb_object']['gb_plane'])

    static_fw = StaticFW(structure=gb, name=name + "_static calc", vasp_input_set=vis_for_static,
                         vasp_input_set_params=static_incar_settings,
                         vasp_cmd=vasp_cmd, prev_calc_dir=prev_calc_dir, db_file=db_file, parents=None)
    fws.append(static_fw)

    wf = Workflow(fws, name="{} gb workflow, e.g., {}".format(len(fws), fws[0].name))
    wf_additional_info = add_additional_fields_to_taskdocs(original_wf=wf, update_dict=additional_info)
    wf_with_tag_info = add_tags(wf_additional_info, tags_list=tag)
    return wf_with_tag_info
