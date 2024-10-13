
# coding: utf-8

# # Qui êtes-vous ?
# <div class="alert alert-block alert-info">
# Écrire ci-dessous le nom, le prénom et le numéro étudiant de chaque membre du binôme :
# </div>

# In[1]:

# 
# 


# # TME 3Y - Régression linéaire multiple

# <div class="alert alert-block alert-warning">
# Dans ce TME, nous allons voir une méthode générale pour effectuer une régression linéaire multiple. Pour cela, nous allons tout d'abord appliquer cette méthode générale pour retrouver les résultats du TME précédent sur la régression simple. Nous allons ensuite utiliser cette nouvelle méthode pour réaliser une régression linéaire multiple.
# </div>
# <div class="alert alert-block alert-warning">
# Voici le plan :
#     <ol>
#         <li>Chargement de données à partir d'un fichier</li>
#         <li>Moindres carrés avec les matrices</li>
#         <li>Tracé du résultat</li>
#         <li>Régression linéaire multiple</li>
#     </ol>
# </div>

# <div class="alert alert-block alert-info">
# Tout d'abord, exécuter la boite ci-dessous pour importer les bibliothèques nécessaires.
# </div>

# In[2]:

import numpy as np
import pandas as pd
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt


# ## 1 - Chargement de données à partir d'un fichier
# 
# <div class="alert alert-block alert-warning">
# Les données nécessaires à ce TME sont stockées dans le fichier <code>advertising.csv</code> qui se présente ainsi :
# <pre>
# TV,Radio,Newspaper,Sales
# 230.1,37.8,69.2,22.1
# 44.5,39.3,45.1,10.4
# 17.2,45.9,69.3,12
# 151.5,41.3,58.5,16.5
# ...
# </pre>
#     Il contient quatre colonnes. Les trois premières sont une mesure de la quantité de publicité diffusée respectivement à la télévision, à la radio et dans les journaux. La dernière colonne mesure les ventes.
#     <br/><em>Source : <a href="https://www.kaggle.com/ashydv/advertising-dataset">https://www.kaggle.com/ashydv/advertising-dataset</a></em>
# </div>
# <div class="alert alert-block alert-info">
#     Q1.1 - Charger les données du fichier dans une variable <code>data_pub</code> à l'aide de la fonction <code>pd.read_csv('nom_de_fichier.csv')</code>.
# </div>

# In[3]:

data_pub = pd.read_csv('advertising.csv')
data_pub


# <div class="alert alert-block alert-info">
#     Q1.2 - Définir la variable <code>x</code> pour qu'elle contienne le vecteur NumPy de la colonne <code>'TV'</code>.
# </div>

# In[4]:

x = data_pub['TV']
x


# <div class="alert alert-block alert-info">
#     Q1.3 - Définir la variable <code>y</code> pour qu'elle contienne le tableau NumPy de la colonne des ventes.
# </div>

# In[5]:

y = np.array(data_pub['Sales'])
y


# ## 2 - Moindres carrés avec les matrices

# <div class="alert alert-block alert-warning">
#     La fonction <code>np.linalg.solve(M, v)</code> permet de déterminer un vecteur $\mathbf{b}$, unique solution lorsqu'elle existe de l'équation matricielle :
#   $$ \mathbf{M} \mathbf{b} = \mathbf{v} $$
#     avec $\mathbf{M}$ matrice carrée $n \times n$ et $\mathbf{v}$ un vecteur de taille $n$.
#   <br/>
#     Cette fonction utilise des outils mathématiques que vous ne verrez que plus tard dans votre cursus. On va se contenter de dire ici qu'elle "fait le travail".
# </div>

# <div class="alert alert-block alert-info">
#   Q2.1 - Résoudre l'équation matricielle $\mathbf{M}\mathbf{b} = \mathbf{v}$ d'inconnue $\mathbf{b}$ où :
#   $$
#   \mathbf{M} = \left[\begin{array}{cc}
#     2 & -1 \\
#     -4 & 3 \\
#   \end{array} \right],
#   \qquad
#   \mathbf{v} = \left[\begin{array}{c} 1 \\ 7 \end{array} \right],
#   \qquad
#   \mathbf{b} = \left[\begin{array}{c} b_0 \\ b_1 \end{array} \right],
#   $$
# Le résultat trouvé doit être l'unique solution du système de deux équations à deux inconnues :
#   $$
#   \left\{
#     \begin{array}{rcrl}
#       2 b_0 & - & b_1 & = 1 \\
#       -4 b_0 & + & 3 b_1 & = 7 \\
#     \end{array}
#   \right.
#   $$
# </div>

# In[6]:

M = np.array([[2, -1], [-4, 3]])
v = np.array([1, 7])
np.linalg.solve(M, v)


# <div class="alert alert-block alert-warning">
#     Dans la partie précédente, vous devez avoir défini les deux séries de données <code>x</code> pour la publicité à la télévision et <code>y</code> pour les ventes.
#  </div>
#  <div class="alert alert-block alert-warning">
#     Nous allons voir une méthode différente du dernier TME pour calculer une régression par la méthode des moindres carrés. On retrouvera évidemment la même droite de régression. Cette nouvelle méthode que nous allons voir sera facilement généralisable pour faire de la régression linéaire multiple dans la partie suivante.<br/>
#     Avec deux séries de données $\mathbf{x}$ et $\mathbf{y}$ puis en définissant :
#       $$
#   \mathbf{X} = [\mathbf{1} \; \mathbf{x}] = \left[\begin{array}{cc}
#                         1 & x_1 \\
#                         \vdots & \vdots \\
#                         1 & x_i \\
#                         \vdots & \vdots \\
#                         1 & x_n \\
#                         \end{array} \right],
#   \qquad
#   \mathbf{b} = \left[\begin{array}{c} b_0 \\ b_1 \end{array} \right],
#   $$
#     D'après le cours, la régression par la méthode des moindres carrés consiste déterminer $\mathbf{b}$ en minimisant la quantité suivante :
#   $$
#   C(a, b) = \|\mathbf{X} \mathbf{b} - \mathbf{y}\|^2$$
#   On obtient alors la droite de régression :
#   $$ y = b_0 + b_1x $$
# </div>
# <div class="alert alert-block alert-warning">
#     Pour minimiser $C(a, b)$, la matrice $X$ n'étant pas carrée, on peut montrer [hors programme] qu'il suffit de trouver $\mathbf{b}$ solution de l'équation matricielle :
#   $$
#   \left(\mathbf{X}^T\mathbf{X}\right) \mathbf{b} = \left(\mathbf{X}^T\mathbf{y}\right)
#   $$
#   (multiplication matricielle par $\mathbf{X}^T$ à gauche)
# </div>

# <div class="alert alert-block alert-info">
# Q2.2 - Quelle instruction faut-il écrire pour créer un vecteur ne contenant que la valeur 1 et de taille 5 ?
# <div>

# In[7]:

np.ones(5)


# <div class="alert alert-block alert-info">
#     Q2.3 - Définir la variable <code>X</code> pour qu'elle corresponde à :
#   $$
#     \mathbf{X} = [\mathbf{1} \; \mathbf{x}] = \left[\begin{array}{cc}
#                         1 & x_1 \\
#                         \vdots & \vdots \\
#                         1 & x_i \\
#                         \vdots & \vdots \\
#                         1 & x_n \\
#                         \end{array} \right].
#   $$
#     On utilisera la fonction <code>np.column_stack(A, B)</code>.
# </div>

# In[8]:

X = np.column_stack([np.ones(len(x)), x])
X


# <div class="alert alert-block alert-warning">
#     Vous devez obtenir :
#     <code>
# array([[  1. , 230.1],
#        [  1. ,  44.5],
#        [  1. ,  17.2],
#        [  1. , 151.5],
#        [  1. , 180.8],
#         ...         ])</code>
#  </div>

# <div class="alert alert-block alert-info">
#     Q2.4 - Calculer $\left(\mathbf{X}^T\mathbf{y}\right)$. Vous devez obtenir un vecteur de taille 2.
# </div>

# In[9]:

np.dot(X.T, y)


# <div class="alert alert-block alert-info">
#     Q2.5 - Retrouver le résultat précédent en calculant les composantes du vecteur :
#     $$
#     \left[\sum_{i = 1}^{n} y_i , \sum_{i = 1}^{n}x_i y_i\right]
#     $$
# </div>

# In[10]:

np.sum(y), np.sum(x*y)


# <div class="alert alert-block alert-info">
#     Q2.6 - Calculer $\left(\mathbf{X}^T\mathbf{X}\right)$. Vous devez obtenir une matrice $2 \times 2$.
# </div>

# In[11]:

np.dot(X.T, X)


# <div class="alert alert-block alert-info">
#     Q2.7 - Retrouver le résultat précédent en calculant les composantes de la matrice :
#   $$
#     \left[\begin{array}{cc}
#                         \displaystyle \sum_{i = 1}^{n}1 & \displaystyle \sum_{i = 1}^{n} x_i \\
#                         \displaystyle \sum_{i = 1}^{n} x_i & \displaystyle \sum_{i = 1}^n x_i^2 \\
#     \end{array} \right]
#   $$  
# </div>

# In[12]:

np.sum(np.ones(len(x))), np.sum(x), np.sum(x), np.sum(x**2)


# <div class="alert alert-block alert-info">
#     Q2.8 - Écrire la définition de la fonction <code>regression_mc</code> qui, étant donné deux séries de nombres <code>x</code> et <code>y</code>, retourne le vecteur $\mathbf{b}$ qui minimise :
#   $$
#   \|\mathbf{X} \mathbf{b} - \mathbf{y} \|^2.
#   $$
#   c'est-à-dire la solution de l'équation matricielle :
#   $$
#     \left(\mathbf{X}^T\mathbf{X}\right) \mathbf{b} = \left(\mathbf{X}^T\mathbf{y}\right)
#   $$
#   avec
#   $$
#     \mathbf{X} = [\mathbf{1} \; \mathbf{x}] = \left[\begin{array}{cc}
#                         1 & x_1 \\
#                         \vdots & \vdots \\
#                         1 & x_i \\
#                         \vdots & \vdots \\
#                         1 & x_n \\
#                         \end{array} \right].
#   $$
#   On utilisera la fonction <code>np.linalg.solve</code> vue plus haut.
# </div>

# In[13]:

def regression_mc(x,y):
    X = np.column_stack([np.ones(len(x)), x])
    return np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))


# <div class="alert alert-block alert-info">
# Q2.9 - Appliquer la fonction <code>regression_mc</code> pour calculer les coefficients de la régression des ventes en fonction de la publicité à la télévision.
# </div>

# In[78]:

ventes = data_pub['Sales']
tv = data_pub['TV']
coeff = regression_mc(tv, ventes)
coeff


# <div class="alert alert-block alert-warning">
#     Vous devez retrouver le résultat du TME précédent, à savoir :
#     <code>
# array([6.97482149, 0.05546477])
# </code>
#     qui correspond à la droite d'équation : $$y = 6.97482149 + 0.05546477 x$$
#  </div>

# <div class="alert alert-block alert-info">
# Exercice facultatif à la maison - Vérifier que les formules, démontrées en cours, des coefficients $b_0$ et $b_1$ de la droite de régression $y = b_0 + b_1x$ sont bien solutions de l'équation matricielle :
#   $$
#       \left(\mathbf{X}^T\mathbf{X}\right) \mathbf{b} = \left(\mathbf{X}^T\mathbf{y}\right)
#   $$
#     c'est-à-dire du système de deux équations à deux inconnues :
#   $$
#   \left\{
#     \begin{array}{rcrl}
#       \displaystyle n b_0 & + & \displaystyle \left( \sum_{i = 1}^{n} x_i \right) b_1 & \displaystyle = \sum_{i = 1}^{n} y_i \\
#       \displaystyle \left( \sum_{i = 1}^{n} x_i \right) b_0 & + & \displaystyle \left( \sum_{i = 1}^n x_i^2 \right) b_1 & \displaystyle = \sum_{i = 1}^{n}x_i y_i \\
#     \end{array}
#   \right.
#   $$
# </div>

# ## 3 - Tracé du résultat

# <div class="alert alert-block alert-info">
# Q3.1 - Écrire la définition de la fonction <code>fig_regression</code> qui, étant donné deux vecteurs d'observations <code>x</code> et <code>y</code> de même taille, génère le  graphique avec le nuage de points et la droite de régression par les moindres carrés en rouge.
# </div>

# In[79]:

def fig_droite(coeff, xmin, xmax, c='red'):
    plt.plot((xmin, xmax), (coeff*xmin + coeff*xmax + coeff), c='R')


# In[32]:

def fig_regression(x,y):
    plt.scatter(x,y)
    


# <div class="alert alert-block alert-info">
# Q3.2 - Appliquer la fonction <code>fig_regression</code> pour visualiser le nuage de points des ventes en fonction de la publicité à la télévision ainsi que la droite de régression sur le même graphique.
# </div>

# In[33]:

fig_regression(tv, ventes)


# ## 4 - Régression linéaire multiple

# <div class="alert alert-block alert-warning">
# Dans cette partie, nous allons utiliser exactement la même méthode que précédemment pour réaliser une régression linéaire multiple.
# <br/>Cette méthode permet de minimiser le critère des moindres carrés pour la régression linéaire multiple vu en cours. Cependant, la démonstration de ce résultat dépasse le cadre de ce cours.
#     
# Nous allons donc rechercher $\mathbf{\tilde{y}} = b_0 + b_1 \mathbf{x_{tv}} + b_2 \mathbf{x_{radio}} + b_3 \mathbf{x_{newspaper}}$ où $\mathbf{\tilde{y}}\approx \mathbf{y}$ au sens des moindres carrés.
# <div>

# <div class="alert alert-block alert-info">
#     Q4.1 - Extraire dans une variable <code>Xmul</code> les trois colonnes <code>'TV'</code>, <code>'Radio'</code> et <code>'Newspaper'</code><!-- dans un tableau <code>numpy</code> à l'aide de la méthode <code>.to_numpy()</code>-->.
# </div>

# In[17]:

Xmul = data_pub[['TV', 'Radio', 'Newspaper']]
Xmul


# <div class="alert alert-block alert-info">
# Q4.2 - Appliquer la fonction <code>regression_mc</code> pour calculer les coefficients $b$ de la régression des ventes en fonction de la publicité à la télévision, à la radio et dans les journaux.
# </div>

# In[107]:

regression_mc(Xmul, ventes)


# <div class="alert alert-block alert-warning">
# Vous devez trouver :
# <code>
# array([4.62512408e+00, 5.44457803e-02, 1.07001228e-01, 3.35657922e-04])
# </code>
# </div>

# <div class="alert alert-block alert-info">
# Q4.3 - Écrire la définition de la fonction <code>fig_regression_mult</code> qui, étant donné une matrice d'observations <code>X</code> et <code>y</code> avec le même nombre de lignes, génère le  graphique avec le nuage de points $(\tilde{y}_i, y_i)$.
# </div>

# In[95]:

def fig_regression_mult(x,y):
    X = np.column_stack([np.ones(len(x)), x])
    y_i = (X * regression_mc(x, y)).sum(axis=1)
    plt.scatter(y, y_i)


# <div class="alert alert-block alert-info">
# Q4.4 - Appliquer la fonction <code>fig_regression_mult</code> pour visualiser le nuage de points des ventes en fonction de $\mathbf{\tilde{y}}$.
# </div>

# In[96]:

fig_regression_mult(Xmul, ventes)


# <div class="alert alert-block alert-info">
# Q4.5 - Calculer le $R^2$ de cette régression multiple.
# </div>

# In[117]:

X = np.column_stack([np.ones(len(Xmul)), Xmul])
y_i = (X * regression_mc(Xmul, y)).sum(axis=1)
np.corrcoef(y_i, y)


# <div class="alert alert-block alert-info">
# Q4.6 - Comparer la contribution de chaque variable de la régression multiple par rapport aux régressions simples vues dans le TME-3X.
# </div>

# In[ ]:



