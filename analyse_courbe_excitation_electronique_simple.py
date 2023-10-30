from outils_analyse.identification_des_pics import determiner_indexes_maximums_scipy
from outils_analyse.lecture_des_fichiers import lire_csv_a_3_colonnes, crop_pour_conserver_que_la_partie_avec_rampe
from outils_analyse.conversion_temps_en_potentiel import \
        calculer_facteur_conversion_temps_en_potentiel_avec_mesure_rampe
from outils_analyse.fits import gaussian_fit, gaus, linear_regression, round_any
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib

# Fixer la taille de la police dans les figures
matplotlib.rcParams.update({'font.size': 18})

# -------------------------------------------------------------------------------------------------
#                        Importer les données et leurs paramètres respectifs 
# -------------------------------------------------------------------------------------------------
# Spécifier les noms des fichiers .csv
csv_file_names = ['Montage_B_step6.2', 'Montage_B_step6.3.1',
                  'Montage_B_step6.3.3_V1.7', 'Montage_B_step6.3.3_V2.3',
                  'Montage_B_step6.3.5_V3.5', 'Montage_B_step6.3.5_V4.5']

# Paramètres ayant été optimisés manuellement pour le tronquage des données et la détection des pics
# de chaque expérience
params_peak_detection = {'zero_threshold': [0.05, 0.05, 0.03, 0.04, 0.03, 0.03],
                         'distance_minimum': [100, 100, 100, 200, 200, 200]}

# Sélectionner une expérience spécifique à analyser parmi celles énumérées
# ci-dessus (0 à 5)
num_exp = 0
csv_file_path = os.path.join("data", csv_file_names[num_exp])

# Importer les données et convertir en array
valeurs_en_array = lire_csv_a_3_colonnes(csv_file_path, 9)

# Afficher les données de tension en fonction du temps
plt.figure()
plt.plot(valeurs_en_array[:, 0],
         valeurs_en_array[:, 2],
         label="Tensions du pico")
plt.plot(valeurs_en_array[:, 0],
         valeurs_en_array[:, 1],
         label="Tensions entre la G1 et le ground")
plt.xlabel("Temps [s]")
plt.ylabel("Tension [V]")
plt.legend()
plt.show()

# -------------------------------------------------------------------------------------------------
#            Retirer les valeurs à l'extérieur de l'activation du générateur de rampe
# -------------------------------------------------------------------------------------------------
# Retrait des valeurs à l'extérieur 
zero_threshold = params_peak_detection['zero_threshold'][num_exp]
valeurs_cropped_debutant_par_t0 = crop_pour_conserver_que_la_partie_avec_rampe(valeurs_en_array,
                                                                               2, zero_threshold, 0.1)
# Réétablir les données tronquées pour débuter à t_0=0
valeurs_cropped_debutant_par_t0[:, 0] -= np.min(valeurs_cropped_debutant_par_t0[:, 0])

# Afficher les données conservées de tension en fonction du temps
plt.figure()
plt.plot(valeurs_cropped_debutant_par_t0[:, 0],
         valeurs_cropped_debutant_par_t0[:, 2],
        label="Tensions du pico")
plt.plot(valeurs_cropped_debutant_par_t0[:, 0],
         valeurs_cropped_debutant_par_t0[:, 1],
        label="Tensions entre la G1 et le ground")
plt.xlabel("Temps [s]")
plt.ylabel("Tension [V]")
plt.legend()
plt.show()

# -------------------------------------------------------------------------------------------------
#          Calculer la pente de la tension du générateur de rampe et son incertitude
# -------------------------------------------------------------------------------------------------
# Calculer la pente de la tension du générateur de rampe et son incertitude.
facteur_valeur, facteur_incertitude = calculer_facteur_conversion_temps_en_potentiel_avec_mesure_rampe(valeurs_cropped_debutant_par_t0,
                                                                                                       0, 2)
print("Pente = ", f"{facteur_valeur} +- {facteur_incertitude}") # Afficher les valeurs

# Convertir les valeurs de temps en valeurs de tension
valeurs_avec_bonnes_unites = valeurs_cropped_debutant_par_t0.copy()
valeurs_avec_bonnes_unites[:, 0] = -facteur_valeur * valeurs_cropped_debutant_par_t0[:, 0]

# Afficher les données de courant mesuré en fonction de la tension entre G1 et le ground
plt.figure()
plt.plot(valeurs_avec_bonnes_unites[:, 0],
         valeurs_avec_bonnes_unites[:, 1],
        label="Courant du pico")
plt.xlabel("Tension entre G1 et le ground [V]")
plt.ylabel("Courant mesuré [nA]")
plt.legend()
plt.show()

# -------------------------------------------------------------------------------------------------
#                 Déterminer les positions approximatives des maximums (pics)
# -------------------------------------------------------------------------------------------------
# Mettre vos données avec les bonnes unités à la place du None
valeurs_avec_bonnes_unites_determination_des_pics = valeurs_avec_bonnes_unites.copy()
distance_minumum = params_peak_detection['distance_minimum'][num_exp] 
liste_des_indexes_des_pics =  determiner_indexes_maximums_scipy(valeurs_avec_bonnes_unites, 1,
                                                                distance_minumum=distance_minumum)
# Obtenir les données pour les indices incluant les pics
valeurs_avec_bonnes_unites_peaks = valeurs_avec_bonnes_unites[liste_des_indexes_des_pics]

print("Estimation des pics:", valeurs_avec_bonnes_unites_determination_des_pics[liste_des_indexes_des_pics,
                                                                                0])
print("Moyenne des écarts:", np.mean(valeurs_avec_bonnes_unites_determination_des_pics[liste_des_indexes_des_pics,
                                                                                       0]))

# Afficher les données du courant du picoampèremètre en fonction de la tension d'accélération
# avec l'identification des maximums détectés
plt.figure()
plt.plot(valeurs_avec_bonnes_unites_determination_des_pics[:, 0],
        valeurs_avec_bonnes_unites_determination_des_pics[:, 1],
        label="Courant du pico")
plt.xlabel("Tension entre G1 et le ground [V]")
plt.scatter(valeurs_avec_bonnes_unites_determination_des_pics[liste_des_indexes_des_pics, 0],
        valeurs_avec_bonnes_unites_determination_des_pics[liste_des_indexes_des_pics, 1],
        label="Estimation des pics")
plt.ylabel("Courant mesuré [nA]")
plt.legend()
plt.show()

# -------------------------------------------------------------------------------------------------
#     Faire une régression gaussienne sur chacun des pics et calculer le potentiel de contact
# -------------------------------------------------------------------------------------------------
# Paramètres qui ont été optimisés manuellement pour les ajustements gaussiens de chaque
# expérience | APPORTER DES MODIFICATIONS POUR OPTIMISER LE PROCESSUS DAVANTAGE!!

# Mettres les paramètres des fits gaussiens pour chaque pic [Amplitude, Moyenne, STD]
peak1_idx_start, peak1_idx_end = liste_des_indexes_des_pics[0] - 50, liste_des_indexes_des_pics[0] + 50
peak1 = gaussian_fit(valeurs_avec_bonnes_unites[peak1_idx_start:peak1_idx_end, 0],
                     valeurs_avec_bonnes_unites[peak1_idx_start:peak1_idx_end, 1],
                     valeurs_avec_bonnes_unites_peaks[0, 1],
                     valeurs_avec_bonnes_unites_peaks[0, 0], 1)

peak2_idx_start, peak2_idx_end = liste_des_indexes_des_pics[1] - 50, liste_des_indexes_des_pics[1] + 50
peak2 = gaussian_fit(valeurs_avec_bonnes_unites[peak2_idx_start:peak2_idx_end, 0],
                     valeurs_avec_bonnes_unites[peak2_idx_start:peak2_idx_end, 1],
                     valeurs_avec_bonnes_unites_peaks[1, 1],
                     valeurs_avec_bonnes_unites_peaks[1, 0], 1)

peak3_idx_start, peak3_idx_end = liste_des_indexes_des_pics[2] - 50, liste_des_indexes_des_pics[2] + 50
peak3 = gaussian_fit(valeurs_avec_bonnes_unites[peak3_idx_start:peak3_idx_end, 0],
                     valeurs_avec_bonnes_unites[peak3_idx_start:peak3_idx_end, 1],
                     valeurs_avec_bonnes_unites_peaks[2, 1],
                     valeurs_avec_bonnes_unites_peaks[2, 0], 1)

peak4_idx_start, peak4_idx_end = liste_des_indexes_des_pics[3] - 50, liste_des_indexes_des_pics[3] + 50
peak4 = gaussian_fit(valeurs_avec_bonnes_unites[peak4_idx_start:peak4_idx_end, 0],
                     valeurs_avec_bonnes_unites[peak4_idx_start:peak4_idx_end, 1],
                     valeurs_avec_bonnes_unites_peaks[3, 1],
                     valeurs_avec_bonnes_unites_peaks[3, 0], 1)


def rounding_peaks(peaks):
      all_values = []
      for i in range(0, 3):
            all_values.append(round_any(peaks[0][i], uncertainty=peaks[1][i]))
      return all_values

print("Pic 1:",
      f"Moyenne: {rounding_peaks(peak1)[1]}",
      f"STD: {rounding_peaks(peak1)[2]}",
      f"Amplitude: {rounding_peaks(peak1)[0]}")
print("Pic 2:",
      f"Moyenne: {rounding_peaks(peak2)[1]}",
      f"STD: {rounding_peaks(peak2)[2]}",
      f"Amplitude: {rounding_peaks(peak2)[0]}")
print("Pic 3:",
      f"Moyenne: {rounding_peaks(peak3)[1]}",
      f"STD: {rounding_peaks(peak3)[2]}",
      f"Amplitude: {rounding_peaks(peak3)[0]}")
print("Pic 4:",
      f"Moyenne: {rounding_peaks(peak4)[1]}",
      f"STD: {rounding_peaks(peak4)[2]}",
      f"Amplitude: {rounding_peaks(peak4)[0]}")


# Afficher les courants en fonction de la tension avec les emplacements approximatifs des maximums
# et ajustements gaussiens
plt.figure(figsize=(16, 10))

plt.plot(valeurs_avec_bonnes_unites[:, 0],
         valeurs_avec_bonnes_unites[:, 1],
         label="Courant mesuré",
         color="black", linewidth=3,
         zorder=1)
plt.scatter(valeurs_avec_bonnes_unites[liste_des_indexes_des_pics, 0],
            valeurs_avec_bonnes_unites[liste_des_indexes_des_pics, 1],
            label="Estimation des pics",
            color="red", marker='o',
            zorder=6)

# Tracer les ajustements gaussiens pour chaque pic
peaks_idx_start = [peak1_idx_start, peak2_idx_start, peak3_idx_start, peak4_idx_start]
peaks_idx_end = [peak1_idx_end, peak2_idx_end, peak3_idx_end, peak4_idx_end]

for i, peak in enumerate([peak1, peak2, peak3, peak4]):
      plt.plot(valeurs_avec_bonnes_unites[peaks_idx_start[i]:peaks_idx_end[i], 0], 
               gaus(valeurs_avec_bonnes_unites[peaks_idx_start[i]:peaks_idx_end[i], 0],
                    peak[0][0], peak[0][1], peak[0][2]),
               label=f"$y = ({rounded_values[0]})\;\exp\{{(x-({rounded_values[1]}))^2\; /\; 2\cdot({rounded_values[2]})^2\}}$",
               color='red', linewidth=2, alpha=0.9)

plt.xlabel("Tension d'accélération [V]")
plt.ylabel("Courant du pico [nA]")
plt.legend(fontsize=14)
plt.tick_params(axis='both', which='both', direction='in')
plt.minorticks_on()

# SAUVEGARDER LA FIGURE DANS LE FICHIER 'FIGURES'
plt.show()
