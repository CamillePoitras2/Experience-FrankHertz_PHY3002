from outils_analyse.identification_des_pics import determiner_indexes_maximums_scipy
from outils_analyse.lecture_des_fichiers import lire_csv_a_3_colonnes, crop_pour_conserver_que_la_partie_avec_rampe
from outils_analyse.conversion_temps_en_potentiel import \
        calculer_facteur_conversion_temps_en_potentiel_avec_mesure_rampe
from outils_analyse.fits import linear_regression
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
csv_file_path = os.path.join("data", "excel", csv_file_names[num_exp])

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
