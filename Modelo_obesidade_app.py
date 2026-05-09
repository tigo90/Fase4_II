import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configuração da página
st.set_page_config(page_title="Preditor de Obesidade - Tech Challenge", layout="wide")

traducao_classes = {
    'Insufficient_Weight': 'Peso Insuficiente',
    'Normal_Weight': 'Peso Normal',
    'Overweight_Level_I': 'Sobrepeso Nível I',
    'Overweight_Level_II': 'Sobrepeso Nível II',
    'Obesity_Type_I': 'Obesidade Tipo I',
    'Obesity_Type_II': 'Obesidade Tipo II',
    'Obesity_Type_III': 'Obesidade Tipo III'
}

@st.cache_resource
def load_artifacts():
    model = joblib.load('modelo_obesidade.pkl')
    le = joblib.load('label_encoder.pkl')
    features = joblib.load('features_columns.pkl')
    return model, le, features

try:
    model, le, features = load_artifacts()
    st.title("🩺 Sistema de Auxílio ao Diagnóstico: Obesidade")
    st.markdown("--- ")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Dados Demográficos")
        gender = st.selectbox("Gênero", ["Masculino", "Feminino"])
        age = st.number_input("Idade", min_value=1, max_value=120, value=25)
        height = st.number_input("Altura (m)", min_value=0.5, max_value=2.5, value=1.70)
        weight = st.number_input("Peso (kg)", min_value=10, max_value=300, value=70)
    with col2:
        st.subheader("Hábitos Alimentares")
        family_history = st.selectbox("Histórico Familiar de Obesidade?", ["Sim", "Não"])
        favc = st.selectbox("Consome comida rica em calorias frequentemente?", ["Sim", "Não"])
        fcvc = st.slider("Frequência de consumo de vegetais (1-3)", 1, 3, 2)
        ncp = st.slider("Número de refeições principais por dia", 1, 4, 3)
        caec = st.selectbox("Consumo de alimentos entre as refeições", ["Às vezes", "Frequentemente", "Sempre", "Não"])
        smoke = st.selectbox("Fumante?", ["Sim", "Não"])
        ch2o = st.slider("Consumo diário de água (litros)", 1, 3, 2)
    with col3:
        st.subheader("Estilo de Vida")
        scc = st.selectbox("Monitora consumo de calorias?", ["Sim", "Não"])
        faf = st.slider("Atividade física semanal (0-3)", 0, 3, 1)
        tue = st.slider("Tempo de uso de eletrônicos (0-2)", 0, 2, 0)
        calc = st.selectbox("Consumo de álcool", ["Não", "Às vezes", "Frequentemente", "Sempre"])
        mtrans = st.selectbox("Meio de transporte principal", ["Transporte Público", "Automóvel", "Caminhada", "Motocicleta", "Bicicleta"])

    if st.button("Realizar Predição"):
        map_sim_nao = {"Sim": "yes", "Não": "no"}
        map_caec = {"Às vezes": "Sometimes", "Frequentemente": "Frequently", "Sempre": "Always", "Não": "no"}
        map_calc = {"Não": "no", "Às vezes": "Sometimes", "Frequentemente": "Frequently", "Sempre": "Always"}
        map_trans = {"Transporte Público": "Public_Transportation", "Automóvel": "Automobile", "Caminhada": "Walking", "Motocicleta": "Motorbike", "Bicicleta": "Bike"}
        map_gender = {"Masculino": "Male", "Feminino": "Female"}

        # Criando DataFrame com as chaves exatas do dataset original antes do dummies
        input_dict = {
            'Gender': map_gender[gender],
            'Age': age,
            'Height': height,
            'Weight': weight,
            'family_history_with_overweight': map_sim_nao[family_history],
            'FAVC': map_sim_nao[favc],
            'FCVC': fcvc,
            'NCP': ncp,
            'CAEC': map_caec[caec],
            'SMOKE': map_sim_nao[smoke],
            'CH2O': ch2o,
            'SCC': map_sim_nao[scc],
            'FAF': faf,
            'TUE': tue,
            'CALC': map_calc[calc],
            'MTRANS': map_trans[mtrans]
        }

        input_df = pd.DataFrame([input_dict])
        
        # Aplicando One-Hot Encoding para alinhar com o treinamento
        input_encoded = pd.get_dummies(input_df)
        
        # Garantindo que todas as colunas do treinamento existam (mesmo que com valor 0)
        for col in features:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        
        # Reordenando as colunas para a ordem exata do modelo
        input_encoded = input_encoded[features]

        # Predição
        prediction = model.predict(input_encoded)
        raw_label = le.inverse_transform(prediction)[0]
        result_label = traducao_classes.get(raw_label, raw_label)

        st.success(f"### Diagnóstico Sugerido: {result_label}")

        # Geração de Relatório
        report_text = f"""RELATÓRIO DE DIAGNÓSTICO PREVENTIVO
-----------------------------------
Dados do Paciente:
- Idade: {age} anos
- Gênero: {gender}
- Peso/Altura: {weight}kg / {height}m

Resultado da Análise:
- Diagnóstico Sugerido: {result_label}

Observações de Hábitos:
- Histórico Familiar: {family_history}
- Atividade Física (0-3): {faf}
- Consumo de Água (1-3): {ch2o}
- Consumo de Vegetais (1-3): {fcvc}
-----------------------------------
Gerado por: Sistema IA Obesidade Tech Challenge
"""
        st.download_button(
            label="📥 Baixar Relatório do Diagnóstico",
            data=report_text,
            file_name=f"relatorio_diagnostico_{age}anos.txt",
            mime="text/plain"
        )

except Exception as e:
    st.error(f"Erro ao processar: {e}")
