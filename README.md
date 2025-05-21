# Projeto de Previsão de Custos Médicos

## Descrição do Projeto
Este Notebook Jupyter contém um projeto de análise de dados que desenvolve um modelo preditivo para estimar custos médicos com base nas características dos pacientes. O modelo utiliza dados rotulados para identificar padrões e relações entre fatores como região geográfica, idade, status de fumante e outros atributos pessoais que influenciam os custos de assistência médica. Esta análise pode auxiliar seguradoras, hospitais e profissionais de saúde a entender melhor os fatores que impactam os custos médicos.

## O que o Código Faz
Este notebook realiza as seguintes operações:
1. Carregamento e pré-processamento de dados
    - Importa dados de pacientes com seus respectivos custos médicos
    - Limpa valores ausentes e trata outliers
    - Codifica variáveis categóricas (como região e status de fumante)
    - Normaliza variáveis numéricas (como idade e IMC)

2. Análise Exploratória de Dados (AED)
    - Gera estatísticas descritivas dos custos médicos por diferentes grupos
    - Cria visualizações para entender a distribuição de custos e sua relação com características dos pacientes
    - Analisa correlações entre variáveis e custos médicos

3. Modelagem Preditiva
    - Divide os dados em conjuntos de treinamento e teste
    - Treina diferentes modelos de regressão (Linear, Random Forest, etc.)
    - Avalia o desempenho dos modelos usando métricas como RMSE e R²
    - Identifica os fatores mais importantes que influenciam os custos médicos

## Instruções de Execução

### Opção 1: Anaconda

1. **Instalar o Anaconda**:
    - Baixe e instale o [Anaconda](https://www.anaconda.com/download) para seu sistema operacional

2. **Iniciar o Jupyter via Anaconda Navigator**:
    - Abra o Anaconda Navigator
    - Clique no botão "Launch" abaixo de Jupyter Notebook

3. **Abrir o Notebook**:
    - Navegue até a localização deste arquivo notebook
    - Clique no arquivo do notebook para abri-lo

4. **Executar o Notebook**:
    - Use o botão "Run" para executar células sequencialmente
    - Alternativamente, use Shift+Enter para executar células uma por uma

### Opção 2: Jupyter (sem Anaconda)

1. **Instalar Python e Jupyter**:

bash
pip install jupyter pandas numpy matplotlib seaborn scikit-learn statsmodels xgboost





2. **Iniciar o Jupyter Notebook**:

bash
jupyter notebook





3. **Abrir e Executar o Notebook**:
    - Navegue até o arquivo do notebook na interface do Jupyter
    - Abra e execute as células como descrito acima

### Opção 3: Google Colab

1. **Acessar o Google Colab**:
    - Acesse [Google Colab](https://colab.research.google.com/)
    - Faça login com sua conta Google

2. **Carregar o Notebook**:
    - Clique em "Arquivo" → "Fazer upload de notebook"
    - Selecione este arquivo notebook do seu computador
    - Alternativamente, se o notebook estiver armazenado no Google Drive, GitHub ou outra plataforma suportada, use a opção de importação apropriada

3. **Executar o Notebook**:
    - Execute as células usando o botão de reprodução ou Shift+Enter
    - Observação: Será necessário carregar o arquivo de dados para o Colab ou conectar ao seu Google Drive onde o arquivo esteja armazenado

## Pacotes Necessários
- pandas: para manipulação e análise de dados
- numpy: para operações numéricas
- matplotlib e seaborn: para visualização de dados
- scikit-learn: para construção e avaliação de modelos preditivos
- statsmodels: para análises estatísticas adicionais
- xgboost: para modelos de gradient boosting

## Arquivos de Dados
- `medical_costs.csv`: Conjunto de dados contendo informações de pacientes e seus custos médicos associados

## Saída Esperada
- Visualizações que mostram relações entre características dos pacientes e custos médicos
- Métricas de desempenho dos modelos preditivos testados
- Um modelo final capaz de prever custos médicos com base nas características do paciente
- Análise dos fatores mais importantes que influenciam os custos médicos

---