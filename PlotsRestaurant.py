# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 13:14:45 2025

@author: geots
"""

def plot_HVAC_results(HVAC_results, day=None, T_low=18, T_ok=22, T_high=26, H_high=70):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    Temp_r1  = HVAC_results['Temp_r1']
    Temp_r2  = HVAC_results['Temp_r2']
    h_r1     = HVAC_results['h_r1']
    h_r2     = HVAC_results['h_r2']
    v        = HVAC_results['v']
    Hum      = HVAC_results['Hum']
    price    = HVAC_results['price']
    Occ_r1   = HVAC_results['Occ_r1']
    Occ_r2   = HVAC_results['Occ_r2']
    Ltemp_r1 = HVAC_results.get('Ltemp_r1', None)
    Ltemp_r2 = HVAC_results.get('Ltemp_r2', None)
    Htemp_r1 = HVAC_results.get('Htemp_r1', None)
    Htemp_r2 = HVAC_results.get('Htemp_r2', None)
    T = list(range(len(Temp_r1)))

    title_suffix = f" — Day {day+1}" if day is not None else ""
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    fig.suptitle(f"HVAC Optimization Results{title_suffix}", fontsize=14, fontweight='bold')

    # --- Room Temperatures ---
    axes[0].plot(T, Temp_r1, label='Room 1', marker='o')
    axes[0].plot(T, Temp_r2, label='Room 2', marker='s')
    axes[0].axhline(T_low,  color='blue',  linestyle='--', alpha=0.5, label=f'T_low ({T_low}°C)')
    axes[0].axhline(T_ok,   color='green', linestyle='--', alpha=0.5, label=f'T_ok ({T_ok}°C)')
    axes[0].axhline(T_high, color='red',   linestyle='--', alpha=0.5, label=f'T_high ({T_high}°C)')

    if Ltemp_r1 and Htemp_r1 and Ltemp_r2 and Htemp_r2:
        for t in T:
            if Ltemp_r1[t] > 0.5 or Ltemp_r2[t] > 0.5:
                axes[0].axvspan(t - 0.5, t + 0.5, color='blue', alpha=0.1)
            if Htemp_r1[t] > 0.5 or Htemp_r2[t] > 0.5:
                axes[0].axvspan(t - 0.5, t + 0.5, color='red', alpha=0.1)
        handles, labels = axes[0].get_legend_handles_labels()
        handles.append(mpatches.Patch(color='blue', alpha=0.3, label='Low temp overrule'))
        handles.append(mpatches.Patch(color='red',  alpha=0.3, label='High temp overrule'))
        axes[0].legend(handles=handles, fontsize=8)
    else:
        axes[0].legend(fontsize=8)  # no overrule data, plain legend

    axes[0].set_ylabel("Temperature (°C)")
    axes[0].set_title("Room Temperatures")
    axes[0].grid(True)

    # --- Heater consumption ---
    axes[1].bar([t - 0.2 for t in T], h_r1, width=0.4, label='Room 1 Heater', alpha=0.7)
    axes[1].bar([t + 0.2 for t in T], h_r2, width=0.4, label='Room 2 Heater', alpha=0.7)
    axes[1].set_ylabel("Heater Power (kW)")
    axes[1].set_title("Heater Consumption")
    axes[1].legend()
    axes[1].grid(True)

    # --- Ventilation and Humidity ---
    for t in T:
        if v[t] > 0.5:
            axes[2].axvspan(t - 0.5, t + 0.5, color='tab:blue', alpha=0.2,
                            label='Ventilation ON' if t == next((i for i in T if v[i] > 0.5), None) else "")
    axes[2].plot(T, Hum, label='Humidity (%)', color='tab:orange', marker='o')
    axes[2].axhline(H_high, color='red', linestyle='--', alpha=0.5, label=f'H_high ({H_high}%)')
    axes[2].set_ylabel("Ventilation / Humidity")
    axes[2].set_title("Ventilation Status and Humidity")
    axes[2].legend()
    axes[2].grid(True)

    # --- Electricity price and occupancy ---
    axes[3].plot(T, price, label='TOU Price (€/kWh)', color='tab:red', marker='x')
    axes[3].bar([t - 0.2 for t in T], Occ_r1, width=0.4, label='Occupancy Room 1', alpha=0.5)
    axes[3].bar([t + 0.2 for t in T], Occ_r2, width=0.4, label='Occupancy Room 2', alpha=0.5)
    axes[3].set_ylabel("Price / Occupancy")
    axes[3].set_xlabel("Hour")
    axes[3].set_xticks(T)
    axes[3].set_title("Electricity Price and Occupancy")
    axes[3].legend()
    axes[3].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()