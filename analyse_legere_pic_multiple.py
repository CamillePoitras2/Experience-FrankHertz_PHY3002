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
# Spécifier le nom du fichier .csv
csv_file_name = 'Montage_B_step6.4'
csv_file_path = os.path.join("data", csv_file_name)
valeurs_en_array = lire_csv_a_3_colonnes(csv_file_path, 9)

# -------------------------------------------------------------------------------------------------
#            Retirer les valeurs à l'extérieur de l'activation du générateur de rampe
# -------------------------------------------------------------------------------------------------
# Retrait des valeurs à l'extérieur 
valeurs_cropped_debutant_par_t0 = crop_pour_conserver_que_la_partie_avec_rampe(valeurs_en_array,
                                                                                2, 0.01, 0.1)
# Réétablir les données tronquées pour débuter à t_0=0
valeurs_cropped_debutant_par_t0[:, 0] -= np.min(valeurs_cropped_debutant_par_t0[:, 0])

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

# -------------------------------------------------------------------------------------------------
#                 Déterminer les positions approximatives des maximums (pics)
# -------------------------------------------------------------------------------------------------
# Mettre vos données avec les bonnes unités à la place du None
valeurs_avec_bonnes_unites_determination_des_pics = valeurs_avec_bonnes_unites.copy()
liste_des_indexes_des_pics =  determiner_indexes_maximums_scipy(valeurs_avec_bonnes_unites, 1,
                                                                hauteur_minimum=0.01,
                                                                distance_minumum=45)
# Obtenir les données pour les indices incluant les pics
valeurs_avec_bonnes_unites_peaks = valeurs_avec_bonnes_unites[liste_des_indexes_des_pics]

print("Estimation des pics:", valeurs_avec_bonnes_unites_determination_des_pics[liste_des_indexes_des_pics,
                                                                                0])

# Afficher les données du courant du pico en fonction de la tension entre G1 et le ground
plt.figure(figsize=(16, 10), dpi=200)
plt.plot(valeurs_avec_bonnes_unites_determination_des_pics[:, 0],
                valeurs_avec_bonnes_unites_determination_des_pics[:, 1],
                label="Courant du pico", c='k')
x_data = valeurs_avec_bonnes_unites_determination_des_pics[liste_des_indexes_des_pics, 0]
y_data = valeurs_avec_bonnes_unites_determination_des_pics[liste_des_indexes_des_pics, 1]
plt.scatter(x_data, y_data,
                label="Estimation des pics", c='red', zorder=10)

# Annoter les pics détectés pour y référer dans la légende
#for i, (x, y) in enumerate(zip(x_data, y_data)):
#        plt.annotate(f"{i+1}", (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

plt.xlabel("Tension entre G1 et le ground [V]")
plt.ylabel("Courant mesuré [nA]")
plt.legend()

#plt.savefig(os.path.join('figures', csv_file_names[num_exp] + "_AnalyseSimpleMultiple.png"), bbox_inches="tight")
plt.show()











