import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Dados de exemplo (substitua pelos seus dados reais)
data = {
    'Texto': [
        "Este filme é incrível!",
        "Que filme horrível!",
        "É um filme mediano.",
        "Eu realmente gostei deste filme.",
        "Não gostei deste filme.",
        "O filme foi bom.",
        "O filme foi ruim.",
        "Este filme é ótimo!",
        "Este filme é péssimo!",
        "O filme é ok."
    ]
}
df = pd.DataFrame(data)

# Simulação de análise de sentimento (substitua por sua própria lógica)
def analisar_sentimento(texto):
    if "incrível" in texto or "gostei" in texto or "ótimo" in texto or "bom" in texto:
        return "Positivo"
    elif "horrível" in texto or "não gostei" in texto or "péssimo" in texto or "ruim" in texto:
        return "Negativo"
    else:
        return "Neutro"

df['Sentimento'] = df['Texto'].apply(analisar_sentimento)


# --- Streamlit app ---
st.title("Análise de Sentimento")

# Exibir o DataFrame
st.write("DataFrame:")
st.dataframe(df)

# Gráfico de pizza
sentimento_counts = df['Sentimento'].value_counts()
fig, ax = plt.subplots()
ax.pie(sentimento_counts, labels=sentimento_counts.index, autopct='%1.1f%%', startangle=90)
ax.axis('equal')  # Equal aspect ratio garante que o gráfico de pizza seja desenhado como um círculo.
st.pyplot(fig)


# Exibir contagens de sentimentos
st.write("Contagens de Sentimentos:")
st.write(sentimento_counts)


# Barra lateral com opções (opcional)
st.sidebar.title("Opções")
# Adicione aqui widgets interativos, como sliders, caixas de seleção, etc.