import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pandas
import plotly_express as plotly
import seaborn as seaborn
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeRegressor

# A variável alvo é 'charges', pois é o que queremos prever.

dados = pandas.read_csv("../../../../Health_insurance.csv")
print(dados.head(10))
print(dados.shape)
print(dados.groupby("charges").describe())
print(dados.info())
print(dados.isnull().sum())

regions = dados["region"]
smokers = dados["smoker"]
print(set(regions))
print(f"Regions count:\n{regions.value_counts()}")
print(f"Smokers count:\n{smokers.value_counts()}")
print(smokers.value_counts())


def create_boxplot_and_histplot(**kwargs):
    """
    Cria um boxplot passando todos os parâmetros diretamente para a função boxplot do seaborn.
    Exemplos de uso: create_boxplot(x="age", data=dados) create_boxplot(x="age", y="charges", hue="smoker", data=dados)
    """
    create_figure()
    seaborn.boxplot(**kwargs)
    create_figure()
    seaborn.histplot(**kwargs)
    pyplot.show()


def create_figure():
    pyplot.figure(figsize=(12, 6))
    pyplot.xticks(rotation=90)


# O parâmetro hue='smoker' nesta chamada de função cria um boxplot separado para cada categoria na coluna 'smoker'.
#
# Nas funções de plotagem do seaborn, o parâmetro hue é usado para adicionar uma variável categórica adicional que será representada por diferentes cores. Isso cria um agrupamento visual que permite comparar distribuições entre múltiplas categorias.
#
# Neste caso específico:
#
#
# O boxplot mostra a relação entre idade (eixo x) e cobranças (eixo y)
# O parâmetro hue='smoker' divide os dados por status de fumante (sim/não)
# Cada grupo de idade terá duas caixas de cores diferentes mostrando a distribuição de cobranças para fumantes vs. não-fumantes
# Isso ajuda a visualizar como as cobranças de seguro variam por idade, com o status de fumante como um fator adicional representado pela cor.
create_boxplot_and_histplot(x='age', y='charges', data=dados, hue='smoker')
create_boxplot_and_histplot(data=dados, x="age")
create_boxplot_and_histplot(data=dados, x="bmi")
create_boxplot_and_histplot(data=dados, x="charges")

plotly.violin(dados, y="charges", x="sex", color="sex", box=True, points="all")

seaborn.set_theme(style="whitegrid", palette="muted")

create_figure()
swarmplot_graph_smoker_charges = seaborn.stripplot(data=dados, x="smoker", y="charges")
swarmplot_graph_smoker_charges.show()

charges = swarmplot_graph_smoker_charges.set(ylabel="")
print(charges)

# Define colunas categóricas para codificação
columns = ['smoker', 'sex', 'region']

# Use um LabelEncoder separado para cada coluna categórica
# Isso previne sobreposição de rótulos entre diferentes variáveis categóricas
encoders = {}
for col in columns:
    le = LabelEncoder()
    dados[col] = le.fit_transform(dados[col])
    encoders[col] = le  # Armazena codificadores para potencial transformação inversa posteriormente

print(dados.head())

# Verificar valores extremos ou ausentes
print("\nVerificando valores ausentes:")
print(dados.isnull().sum())

print("\nEstatísticas descritivas para identificar valores extremos:")
print(dados.describe())

# Verificar valores infinitos ou NaN
print("\nVerificando valores infinitos ou NaN:")
print(np.isinf(dados).sum())
print(np.isnan(dados).sum())

# Criar matriz de correlação para entender relações entre características
correlation_matrix = dados.corr().round(2)
fig, ax = pyplot.subplots(figsize=(11,11))
seaborn.heatmap(data=correlation_matrix, annot=True, linewidths=.5, ax=ax)

# Dividir dados em variáveis (X) e alvo (y)
X = dados.drop('charges', axis=1)
y = dados['charges']

# Aplicando transformação logarítmica ao alvo para lidar com skewness (se necessário)
# Isso pode ajudar a estabilizar a variância e melhorar o desempenho do modelo
y_log = np.log1p(y)  # log(1+x) para lidar com valores zero

# Dividir em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Também dividir a versão logarítmica do alvo
_, _, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Escalonar características para melhor desempenho do modelo
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Adicionar pequena constante para evitar divisão por zero em matrizes
epsilon = 1e-8
X_train_scaled = np.where(abs(X_train_scaled) < epsilon, epsilon, X_train_scaled)
X_test_scaled = np.where(abs(X_test_scaled) < epsilon, epsilon, X_test_scaled)

# Criar dicionário de modelos para avaliar
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'KNN': KNeighborsRegressor(),
    'Decision Tree': DecisionTreeRegressor(random_state=42)
}

# Treina e avalia cada modelo
results = {}
for name, model in models.items():
    # Treina o modelo
    model.fit(X_train_scaled, y_train)

    # Faz previsões
    y_pred = model.predict(X_test_scaled)

    # Calcula métricas
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Armazena os resultados
    results[name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

    # Imprime resultados individuais do modelo em PT-BR
    print(f"\nResultados do modelo {name}:")
    print(f"Erro Médio Absoluto (MAE): R$ {mae:.2f}")
    print(f"Erro Quadrático Médio (MSE): R$ {mse:.2f}")
    print(f"Raiz do Erro Quadrático Médio (RMSE): R$ {rmse:.2f}")
    print(f"Coeficiente de Determinação (R²): {r2:.4f}")

# Mostra resumo de todos os modelos para comparação em PT-BR
print("\nResumo Comparativo dos Modelos:")
print("=" * 50)
for name, metrics in results.items():
    print(f"{name}: MAE=R$ {metrics['MAE']:.2f}, RMSE=R$ {metrics['RMSE']:.2f}, R²={metrics['R2']:.4f}")

# Encontra o melhor modelo baseado no R2
best_model_name = max(results.keys(), key=lambda k: results[k]['R2'])
print(f"\nMelhor modelo baseado no R²: {best_model_name} com R²={results[best_model_name]['R2']:.4f}")

# Visualiza valores reais vs previstos para o melhor modelo
best_model = models[best_model_name]
y_pred = best_model.predict(X_test_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Custos Reais')
plt.ylabel('Custos Previstos')
plt.title(f'{best_model_name}: Custos Reais vs Previstos')
plt.tight_layout()
plt.show()

# Importância das variáveis para o modelo Random Forest (se for o melhor ou um dos melhores modelos)
if 'Random Forest' in models:
    rf_model = models['Random Forest']
    feature_importance = pandas.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    feature_importance.plot(kind='bar')
    plt.title('Importância das Características (Random Forest)')
    plt.xlabel('Características')
    plt.ylabel('Importância')
    plt.tight_layout()
    plt.show()

    print("\nImportância das Características:")
    for feature, importance in feature_importance.items():
        print(f"{feature}: {importance:.4f}")

# Exemplo: Prever custos para um novo paciente
# Criando um paciente de exemplo - 40 anos

# Exemplo: Prever custos para um novo paciente
# Criando um paciente de exemplo - 40 anos, masculino (1), IMC 25, 2 filhos, não fumante (0), da região sudeste (2)
new_patient = np.array([[40, 1, 25, 2, 0, 2]])
new_patient_scaled = scaler.transform(new_patient)
predicted_charge = best_model.predict(new_patient_scaled)[0]
print(f"Custo previsto para um homem de 40 anos com IMC 25, 2 filhos, não fumante, da região sudeste: R$ {predicted_charge:.2f}")

