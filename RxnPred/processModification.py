# -*- coding: utf-8 -*-
# @Time: 2023/8/21
# @Author: Haosong Zhang
# @Mail: zhanghs@sioc.ac.cn
# @FileName: processModification.py

"""
modification to formula
"""

# import torch
import re
from molmass import Formula
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

periodic_table = [
    "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
    "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca",
    "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt",
    # "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]
atom_mass_dict = {
    x: Formula(x).monoisotopic_mass
    for x in periodic_table
}


def formula2dict(formula):
    element_count = {}
    i = 0

    while i < len(formula):
        if formula[i].isalpha():
            element = formula[i]  # get element
            # add lower to element
            i += 1
            while i < len(formula) and formula[i].islower():
                element += formula[i]
                i += 1
            # get digit
            count = 0
            while i < len(formula) and formula[i].isdigit():
                count = count * 10 + int(formula[i])
                i += 1
            # no-digit means 1
            if count == 0:
                count = 1
            # add to dict
            if not is_valid_element(element):
                print(f"Warning: Element {element} is not in the periodic table.")
            if element in element_count:
                element_count[element] += count
            else:
                element_count[element] = count
        else:
            i += 1  # skip

    element_count = filter_non_elements(element_count)
    element_count = sort_elements_by_periodic_table(element_count)
    return element_count


def modification2dict(modification):
    element_count = {}
    current_element = ""
    current_count = 0
    is_positive = True

    for char in modification:
        # start with "<"
        if char == "<":
            is_positive = True  # default
            current_element = ""
            current_count = 0
            continue
        # digit or alpha?
        if char.isdigit():
            current_count = current_count * 10 + int(char)
            continue
        if char.isalpha():
            current_element += char
            if current_count == 0:
                current_count = 1
            continue
        # update now
        if current_element:
            if not is_valid_element(current_element):
                print(f"Warning: Element {current_element} is not in the periodic table.")
            if is_positive:
                element_count[current_element] = element_count.get(current_element, 0) + current_count
            else:
                element_count[current_element] = element_count.get(current_element, 0) - current_count
        # check direction "-" or "+" for next update, default "+"
        if char == "-":
            is_positive = False
        if char == "+":
            is_positive = True
        # reset
        current_element = ""
        current_count = 0
        # end with ">"
        if char == ">":
            break

    element_count = filter_non_elements(element_count)
    return element_count


def sort_elements_by_periodic_table(element_count):
    sorted_elements = sorted(element_count.items(), key=lambda x: periodic_table.index(x[0]))
    sorted_element_count = dict(sorted_elements)
    return sorted_element_count


def is_valid_element(element):
    return element in periodic_table


def filter_non_elements(element_count):
    valid_elements = {}
    for element in element_count:
        if element in periodic_table:
            valid_elements[element] = element_count[element]
    return valid_elements


def combine_dicts(dict1, dict2, direction=True):
    combined_dict = {}
    if direction:
        for key, value in dict2.items():
            if key in dict1:
                dict1[key] += value
            else:
                dict1[key] = value
    else:
        for key, value in dict2.items():
            if key in dict1:
                dict1[key] -= value
            else:
                dict1[key] = -value
    for key, value in dict1.items():
        if value == 0:
            continue
        combined_dict[key] = value
    return combined_dict


def dict2formula(element_count):
    formula = ""
    for element, count in element_count.items():
        if count < 1:
            print(f"Warning: Element {element} is less than 1, removing.")
            continue
        formula += element
        if count != 1:
            formula += str(abs(count))
    return formula


def getAtomDiffByFormula(formula1, formula2):
    formula_dict1 = formula2dict(formula1)
    formula_dict2 = formula2dict(formula2)
    return getAtomDiff(formula_dict1, formula_dict2)


def getAtomDiff(atom_dict1, atom_dict2):
    final_dict = combine_dicts(atom_dict1, atom_dict2, direction=False)
    total_mass = sum(atom_mass_dict.get(atom) * final_dict[atom] for atom in final_dict)
    if total_mass < 0:
        final_dict = combine_dicts({}, final_dict, direction=False)
    final_dict = sort_elements_by_periodic_table(final_dict)
    return getModification(final_dict)


def getModification(atom_dict):
    string = "<"
    for element, value in atom_dict.items():
        if value < 0:
            if value == -1:
                string += "-" + element
            else:
                string += "-" + str(abs(value)) + element
        elif value > 0:
            if value == 1:
                string += "+" + element
            else:
                string += "+" + str(value) + element
    string += ">"
    string = re.sub(r'<\+', r'<', string)
    return string


if __name__ == '__main__':

    # formula = "C16H22FNO"
    # formula_dict = formula2dict(formula)
    # print(formula_dict)  # {'H': 22, 'C': 16, 'N': 1, 'O': 1, 'F': 1}
    #
    # special_formula = "<H+3P+O>"
    # modification_dict = modification2dict(special_formula)
    # print(modification_dict)  # {'H': 1, 'P': 3, 'O': 1}
    # special_formula = "<-H-3S-O+P>"
    # modification_dict = modification2dict(special_formula)
    # print(modification_dict)  # {'H': -1, 'S': -3, 'O': -1, 'P': 1}
    # special_formula = "<-H-3S-55O+Gw+U>"
    # modification_dict = modification2dict(special_formula)
    # print(modification_dict)  # Warning: Element Gw is not in the periodic table.
    # # {'H': -1, 'S': -3, 'O': -55, 'U': 1}
    #
    # combined_dict = combine_dicts(formula_dict, modification_dict)
    # print(combined_dict)  # {'H': 21, 'C': 16, 'N': 1, 'O': -54, 'F': 1, 'S': -3, 'U': 1}
    #
    # formula = dict2formula(combined_dict)
    # print(formula)  # 'H21C16NO54FS3U'

    formula = CalcMolFormula(Chem.MolFromSmiles("C1=NC2=C(C(=N1)N)N=CN2[C@H]3[C@@H]([C@@H]([C@H](O3)COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O)O"))
    formula_output = CalcMolFormula(Chem.MolFromSmiles("C1=NC2=C(C(=N1)N)N=CN2[C@H]3[C@@H]([C@@H]([C@H](O3)COP(=O)(O)OP(=O)(O)O)O)O"))
    print(formula)  # C10H16N5O13P3
    formula_dict = formula2dict(formula)
    print(formula_dict)  # {'H': 16, 'C': 10, 'N': 5, 'O': 13, 'P': 3}

    modification = "<-H-3O-P>"
    print(modification)  # <-H-3O-P>
    modification_dict = modification2dict(modification)
    print(modification_dict)  # {'H': -1, 'O': -3, 'P': -1}

    combined_dict = combine_dicts(formula_dict, modification_dict)
    print(combined_dict)  # {'H': 15, 'C': 10, 'N': 5, 'O': 10, 'P': 2}

    formula = dict2formula(combined_dict)
    print(formula)  # H15C10N5O10P2
    print(formula_output)  # C10H15N5O10P2
