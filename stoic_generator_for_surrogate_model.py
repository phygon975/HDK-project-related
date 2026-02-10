import numpy as np
import os
import re
import matplotlib.pyplot as plt

# ==============================================================================
# 0. User Configuration
# ==============================================================================
# Gambaro et al. (2011)은 C70까지 다룹니다.
MAX_C_NUM = 70  
OUTPUT_FILENAME = f"Gambaro2011_Model1_C{MAX_C_NUM}_Kinetic_adjusted_iter633.txt"

# ==============================================================================
# 1. Parameter Definitions (Sun et al. 2017, Table 2)
# ==============================================================================
# 단위 정보:
# E (Activation Energy): J/mol (Aspen 입력 단위인 kJ/kmol과 수치 동일)
# k (Pre-exponential): kmol/kg-h (기본 단위)
# K_L (Adsorption): MPa^-1 (코드 내부에서 Pa^-1로 환산됨)

par = {
    # --- Adsorption Constants (KL) ---
    # n-Paraffin: 0.1 * Par1 * (tanh(Par2*i - 2) + 1)
    1: 1.12e+3,  2: 2.96e-1,
    # iso-Paraffin: 0.1 * Par3 * (tanh(Par4*i - 2) + 1)
    3: 1.43e0,  4: 6.38e-2,
    
    # --- Activation Energies (E) [J/mol] ---
    # Isomerization: Par5 * i^Par6
    5: 5.16e4,  6: 4.05e-1,
    # Cracking: Par7 * i^Par8
    7: 4.49e4,  8: 5.41e-1,    
    
    # --- Frequency Factors (k0) [mol/(g-h)] ---
    # Isomerization: Par9 * i^Par10 
    # (Note: Par9 inferred as 2.98e-4 based on Table 1 layout & user estimate)
    9: 5.27e-08, 10: 6.35e0,
    # Cracking: Par11 * i^Par12
    11: 8.05e-10, 12: 7.21e0, 
    
    # --- Equilibrium & Stoichiometry ---
    # Keq (Isom): Par13*(i^2-9) + Par14*(i-3)
    13: 1.23e2, 14: 2.16e3,
    # Iso-fraction in cracking products (Par15) - Model 1 Value
    15: 0.878 
}

# Reference Temperature (Gambaro: 632.15 K = 359.0 C)
T_REF_C = 359.0 

# ==============================================================================
# 2. Helper Functions
# ==============================================================================
def get_comp_name(n, isomer=False):
    if n == 1: return "CH4"
    if n == 2: return "C2H6"
    if n == 3: return "C3H8"
    prefix = "ISO-" if isomer else "N-"
    return f"{prefix}C{n}"

def get_carbon_num(comp_name):
    if comp_name == "CH4": return 1
    if comp_name == "C2H6": return 2
    if comp_name == "C3H8": return 3
    match = re.search(r'C(\d+)', comp_name)
    if match: return int(match.group(1))
    return 0

def calc_KL(i, isomer=False):
    # 논문 Eq.18, Eq.20 (0.1 계수 포함) [cite: 187, 190]
    # 단위 변환: MPa^-1 -> Pa^-1 (* 1e-6)
    if not isomer:
        val = 0.1 * par[1] * (np.tanh(i * par[2] - 2) + 1)
    else:
        val = 0.1 * par[3] * (np.tanh(i * par[4] - 2) + 1)
    return val * 1e-6

def calc_Keq(i):
    # Eq.26: Quadratic function for Keq
    return (par[13] * (i**2 - 9) + par[14] * (i - 3) )

def get_kinetics(i, type="isom"):
    if type == "isom":
        k0_raw = par[9] * (i ** par[10])
        E_val = par[5] * (i ** par[6])
    else:
        k0_raw = par[11] * (i ** par[12])
        E_val = par[7] * (i ** par[8])
    return k0_raw, E_val

def format_with_slash_wrap(prefix, items, wrap_limit=72, indent="        "):
    lines = []
    current_line = prefix
    first_item = True
    for item in items:
        if not first_item:
            text_to_add = " / " + item
        else:
            text_to_add = " " + item
        if len(current_line) + len(text_to_add) > wrap_limit:
            if not first_item:
                current_line += " / &"
                lines.append(current_line)
                current_line = indent + item 
            else:
                current_line += " " + item + " &"
                lines.append(current_line)
                current_line = indent
        else:
            current_line += text_to_add
        first_item = False
    if current_line.strip():
        lines.append(current_line)
    return lines

def create_g_ads_exp_block(total_terms):
    lines = []
    lines.append("    G-ADS-EXP &")
    comp_list = []
    # [Logic Preserved] Adding H2 with exponent 1 to all terms
    # This effectively multiplies the entire denominator by H2 concentration.
    comp_list.append(("H2", 1))
    curr = 2
    for i in range(1, MAX_C_NUM + 1):
        comp_list.append((get_comp_name(i, False), curr))
        curr += 1
    for i in range(4, MAX_C_NUM + 1):
        comp_list.append((get_comp_name(i, True), curr))
        curr += 1

    for cid, term_idx in comp_list:
        head = f"        GLHHWNO=1 CID={cid} SSID=MIXED EXPONENT="
        current_l = head
        for t in range(1, total_terms + 1):
            if cid == "H2": val = "1."
            else: val = "1." if t == term_idx else "0."
            if len(current_l) + len(val) + 1 > 72:
                lines.append(current_l + " &")
                current_l = "        " + val
            else:
                current_l += " " + val
        if len(current_l) + 4 > 72:
             lines.append(current_l + " &")
             lines.append("        / &") 
        else:
             lines.append(current_l + " / &")
    if lines:
        if lines[-1].endswith(" / &"): lines[-1] = lines[-1][:-4]
        elif lines[-1].strip() == "/ &": lines.pop()
    return lines

def plot_model_parameters():
    print("\\n[Graph Generation] Plotting kinetic parameters...")
    # 1. 탄소수 배열 생성
    C_range = np.arange(1, MAX_C_NUM + 1)
    
    # K_L 값 계산 (n-Paraffin, iso-Paraffin)
    # Aspen Code uses: val [MPa^-1] * 1e-6 = Output [Pa^-1]
    # So: Aspen Value = Paper Value * 1e-6
    # Or: Paper Value = Aspen Value * 1e6
    KL_n_aspen = np.array([calc_KL(i, False) for i in C_range]) # Pa^-1
    KL_iso_aspen = np.array([calc_KL(i, True) for i in C_range]) # Pa^-1
    
    KL_n_paper = KL_n_aspen * 1e6 # MPa^-1
    KL_iso_paper = KL_iso_aspen * 1e6 # MPa^-1
    
    # E 값 계산 (Isom, Cracking)
    # Code: J/mol (numerically equal to kJ/kmol)
    E_isom = np.array([get_kinetics(i, "isom")[1] for i in C_range])
    E_ck_full = np.array([get_kinetics(i, "ck")[1] for i in C_range])
    
    # k0 값 계산
    # Code: kmol/kg-h (numerically equal to mol/g-h)
    k0_isom = np.array([get_kinetics(i, "isom")[0] for i in C_range])
    k0_ck_full = np.array([get_kinetics(i, "ck")[0] for i in C_range])

    # # -------------------------------------------------------
    # # Plot 1: Adsorption Constants (KL) - Dual Axis
    # # -------------------------------------------------------
    # fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # ax1.set_title(f"Adsorption Constants ($K_L$) Comparison\\nGambaro Model 1 (C{MAX_C_NUM})")
    # ax1.set_xlabel("Carbon Number")
    # ax1.set_ylabel("Paper Value ($MPa^{-1}$)", color='tab:blue', fontsize=12, fontweight='bold')
    
    # # Plot Paper Values (Points from paper style)
    # ln1 = ax1.plot(C_range, KL_n_paper, 'D', label='n-Paraffin (Paper Unit)', color='tab:blue', markerfacecolor='none', markersize=6)
    # ln2 = ax1.plot(C_range, KL_iso_paper, 's', label='iso-Paraffin (Paper Unit)', color='navy', markerfacecolor='none', markersize=6)
    
    # ax1.tick_params(axis='y', labelcolor='tab:blue')
    # ax1.grid(True, which='both', linestyle='--', alpha=0.5)

    # # Secondary Axis for Aspen Values
    # ax2 = ax1.twinx()
    # ax2.set_ylabel("Aspen Input Value ($Pa^{-1}$)", color='tab:red', fontsize=12, fontweight='bold')
    
    # # Plot Aspen Values (Lines)
    # # They should overlap perfectly if the factor 1e-6 is correct.
    # # We use a slightly transparent line to show the overlap.
    # ln3 = ax2.plot(C_range, KL_n_aspen, '-', label='n-Paraffin (Aspen Unit)', color='tab:red', alpha=0.5, linewidth=2)
    # ln4 = ax2.plot(C_range, KL_iso_aspen, '--', label='iso-Paraffin (Aspen Unit)', color='darkred', alpha=0.5, linewidth=2)
    
    # ax2.tick_params(axis='y', labelcolor='tab:red')

    # # Legends
    # lns = ln1 + ln2 + ln3 + ln4
    # labs = [l.get_label() for l in lns]
    # ax1.legend(lns, labs, loc='center right')
    
    # plt.tight_layout()
    # plt.savefig("Gambaro_KL_Comparison.png", dpi=300)
    # plt.close()
    # print("✅ Generated Plot: Gambaro_KL_Comparison.png")

    # # -------------------------------------------------------
    # # Plot 2: Activation Energy (E)
    # # -------------------------------------------------------
    # plt.figure(figsize=(10, 6))
    # plt.title("Activation Energy ($E$)\\nPaper (J/mol) = Aspen (kJ/kmol)")
    # plt.plot(C_range, E_isom, 'b-', label='Isomerization', linewidth=2)
    # plt.plot(C_range, E_ck_full, 'r--', label='Cracking', linewidth=2)
    # plt.xlabel("Carbon Number")
    # plt.ylabel("Activation Energy")
    # plt.legend()
    # plt.grid(True, linestyle='--', alpha=0.5)
    # plt.tight_layout()
    # plt.savefig("Gambaro_E_Comparison.png", dpi=300)
    # plt.close()
    # print("✅ Generated Plot: Gambaro_E_Comparison.png")

    # # -------------------------------------------------------
    # # Plot 3: Pre-exponential Factor (k0)
    # # -------------------------------------------------------
    # plt.figure(figsize=(10, 6))
    # plt.title("Frequency Factor ($k_0$)\\nPaper (mol/g-h) ≈ Aspen (kmol/kg-h)")
    # plt.plot(C_range, k0_isom, 'b-', label='Isomerization', linewidth=2)
    # plt.plot(C_range, k0_ck_full, 'r--', label='Cracking', linewidth=2)
    # plt.xlabel("Carbon Number")
    # plt.ylabel("Pre-exponential Factor ($k_0$)")
    # plt.yscale('log')
    # plt.legend()
    # plt.grid(True, linestyle='--', alpha=0.5)
    # plt.tight_layout()
    # plt.savefig("Gambaro_k0_Comparison.png", dpi=300)
    # plt.close()
    # print("✅ Generated Plot: Gambaro_k0_Comparison.png")

    # # -------------------------------------------------------
    # # Plot 3-2: Pre-exponential Factor (k0) - Linear Scale
    # # -------------------------------------------------------
    # plt.figure(figsize=(10, 6))
    # plt.title("Frequency Factor ($k_0$) - Linear Scale\\nPaper (mol/g-h) ≈ Aspen (kmol/kg-h)")
    # plt.plot(C_range, k0_isom, 'b-', label='Isomerization', linewidth=2)
    # plt.plot(C_range, k0_ck_full, 'r--', label='Cracking', linewidth=2)
    # plt.xlabel("Carbon Number")
    # plt.ylabel("Pre-exponential Factor ($k_0$)")
    # # plt.yscale('log') # Linear scale
    # plt.legend()
    # plt.grid(True, linestyle='--', alpha=0.5)
    # plt.tight_layout()
    # plt.savefig("Gambaro_k0_Comparison_Linear.png", dpi=300)
    # plt.close()
    # print("✅ Generated Plot: Gambaro_k0_Comparison_Linear.png")

# ==============================================================================
# 3. Data Preparation
# ==============================================================================
terms_db = []
terms_db.append("GLHHWNO=1 TERM=1 A=0.0") 
curr_term = 2
for i in range(1, MAX_C_NUM + 1):
    val = np.log(calc_KL(i, False))
    terms_db.append(f"GLHHWNO=1 TERM={curr_term} A={val:.5f}")
    curr_term += 1
for i in range(4, MAX_C_NUM + 1):
    val = np.log(calc_KL(i, True))
    terms_db.append(f"GLHHWNO=1 TERM={curr_term} A={val:.5f}")
    curr_term += 1
total_terms = curr_term - 1

reactions_db = []
rxn_counter = 1 

for i in range(5, MAX_C_NUM + 1):
    # Isomerization
    k0_raw, E_val = get_kinetics(i, "isom")
    ln_Keq = -np.log(calc_Keq(i))
    
    rxn_iso = {
        'id': rxn_counter, 'type': 'ISOM', 'name': f"ISOM-{i}", 
        'rev': 'YES', 'k0': k0_raw, 'E': E_val,
        'stoic': [f"{get_comp_name(i, False)} -1.", f"{get_comp_name(i, True)} 1."],
        'comp_n': get_comp_name(i, False), 'comp_iso': get_comp_name(i, True), 
        'ln_inv_Keq': ln_Keq
    }
    reactions_db.append(rxn_iso)
    rxn_counter += 1

    # Cracking
    if i >= 6:
        k0_raw, E_val = get_kinetics(i, "ck")
        stoic_list = [f"{get_comp_name(i, True)} -1.", "H2 -1."]
        
        temp_products = {} 
        par15 = par[15]
        
        # [Stoichiometry Logic Preserved]
        if i == 6:
            temp_products["C3H8"] = 2.0
        elif i == 7:
            # C7 -> C3 + C4 logic handled by generic loop or manually here if empty range
            # Range(4, 4) is empty loop.
            # p = 1.0 / (7-6) = 1.0. 
            # R_mol = 2.0. R_carb = 7.0.
            # n_B (C4) = (7 - 3*2)/1 = 1.0. n_A (C3) = 1.0.
            # So the generic solver below works perfectly for i=7 too.
            pass
            
        # Generic Logic
        middle_mol_sum = 0.0
        middle_carb_sum = 0.0
        
        p = 1.0 / (i - 6.0) if i > 6 else 0.0
        
        if i > 6:
            for k in range(4, i - 3): 
                coeff_theoretical = 2.0 * p
                c_iso = get_comp_name(k, True)
                c_n = get_comp_name(k, False)
                
                amount_iso = round(coeff_theoretical * par15, 10)
                amount_n   = round(coeff_theoretical * (1 - par15), 10)
                
                if amount_iso > 1e-12:
                    temp_products[c_iso] = amount_iso
                    middle_mol_sum += amount_iso
                    middle_carb_sum += amount_iso * k
                if amount_n > 1e-12:
                    temp_products[c_n] = amount_n
                    middle_mol_sum += amount_n
                    middle_carb_sum += amount_n * k

            R_mol = 2.0 - middle_mol_sum
            R_carb = float(i) - middle_carb_sum
            
            if (i - 6) > 0:
                n_B = (R_carb - 3.0 * R_mol) / (i - 6.0)
                n_A = R_mol - n_B
                
                n_B = round(n_B, 10)
                n_A = round(n_A, 10)
                
                # C3
                temp_products["C3H8"] = n_A
                
                # Cn-3
                carbon_nm3 = i - 3
                if carbon_nm3 > 3:
                    c_iso_B = get_comp_name(carbon_nm3, True)
                    c_n_B   = get_comp_name(carbon_nm3, False)
                    amt_iso_B = round(n_B * par15, 10)
                    amt_n_B   = round(n_B - amt_iso_B, 10) # Balance closure
                    
                    temp_products[c_iso_B] = temp_products.get(c_iso_B, 0) + amt_iso_B
                    temp_products[c_n_B]   = temp_products.get(c_n_B, 0) + amt_n_B

        for c, amt in temp_products.items():
            if amt > 1e-12: 
                stoic_list.append(f"{c} {amt:.10f}")

        rxn_ck = {
            'id': rxn_counter, 'type': 'CK', 'name': f"CK-{i}", 
            'rev': 'NO', 'k0': k0_raw, 'E': E_val,
            'stoic': stoic_list,
            'comp_iso': get_comp_name(i, True)
        }
        reactions_db.append(rxn_ck)
        rxn_counter += 1

# ==============================================================================
# 4. Line Generation
# ==============================================================================
lines = []
lines.append("REACTIONS HDK GENERAL")

for r in reactions_db:
    l = f"    REAC-DATA {r['id']} NAME={r['name']} REAC-CLASS=GLHHW PHASE=V &"
    lines.append(l)
    if r['rev'] == 'YES':
        lines.append(f"        CBASIS=FUGACITY RBASIS=CAT-WT REVERSIBLE=YES &")
        lines.append(f"        REV-METH=USER-SPEC RATE-UNITC=\"KMOL/KG-HR\"")
    else:
        lines.append(f"        CBASIS=FUGACITY RBASIS=CAT-WT REVERSIBLE=NO &")
        lines.append(f"        RATE-UNITC=\"KMOL/KG-HR\"")

for r in reactions_db:
    # E values in Sun et al. are J/mol. Aspen <kJ/kmol> is numerically same.
    lines.append(f"    RATE-CON {r['id']} PRE-EXP={r['k0']:.6e} ACT-ENERGY={r['E']:.2f} <kJ/kmol> &")
    lines.append(f"        T-REF={T_REF_C}")

for r in reactions_db:
    prefix = f"    STOIC {r['id']} MIXED"
    wrapped = format_with_slash_wrap(prefix, r['stoic'], wrap_limit=70)
    lines.extend(wrapped)

for r in reactions_db:
    if r['type'] == 'ISOM':
        lines.append(f"    DFORCE-EXP {r['id']} MIXED {r['comp_n']} 1.")
    elif r['type'] == 'CK':
        lines.append(f"    DFORCE-EXP {r['id']} MIXED {r['comp_iso']} 1.")

for r in reactions_db:
    if r['type'] == 'ISOM':
        lines.append(f"    DFORCE-EXP-2 {r['id']} MIXED {r['comp_iso']} 1.")

eq1_items = []
eq2_items = []
for r in reactions_db:
    if r['type'] == 'ISOM':
        eq1_items.append(f"REACNO={r['id']} A=0.")
        eq2_items.append(f"REACNO={r['id']} A={r['ln_inv_Keq']:.5f}")

lines.extend(format_with_slash_wrap("    DFORCE-EQ-1", eq1_items, wrap_limit=72))
lines.extend(format_with_slash_wrap("    DFORCE-EQ-2", eq2_items, wrap_limit=72))

lines.extend(create_g_ads_exp_block(total_terms))
lines.append(f"    G-ADS-NTERM GLHHWNO=1 ADS-NTERM={total_terms}")
lines.extend(format_with_slash_wrap("    G-ADS-EQ", terms_db, wrap_limit=72))
lines.append("    G-ADS-POW GLHHWNO=1 EXPONENT=1.")

act_entries = [str(r['id']) for r in reactions_db]
lines.extend(format_with_slash_wrap("    REAC-ACT", act_entries, wrap_limit=72))

# ==============================================================================
# 5. File Output
# ==============================================================================
try:
    with open(OUTPUT_FILENAME, "w") as f:
        f.write("\n".join(lines))
    print("=" * 60)
    print(f"✅ 파일 생성 완료: {OUTPUT_FILENAME}")
    print(f"✅ 적용된 파라미터: Gambaro et al. (2011) Table 1")
    print(f"✅ 탄소수 범위: C1 ~ C{MAX_C_NUM}")
    print("=" * 60)
    plot_model_parameters()

except Exception as e:
    print(f"Error: {e}")