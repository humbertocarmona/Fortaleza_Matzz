# %%
df = pd.read_csv("./data/perfil_eleitor_secao_2024_CE.csv", sep=";")
# %%  --------------------------------------------------------------------------
dff = df[df["NM_MUNICIPIO"] == "FORTALEZA"]
age_voters = dff.groupby("DS_FAIXA_ETARIA").size().reset_index(name="Count")



age_voters["f"] = [0, 0, 1, 1, 2, 3, 4, 5, 6, 0, 6]
# age_voters = age_voters[age_voters["f"] > 0]



age_category_mapping = {
    '16 anos': '16 anos',
    '17 anos': '17 anos',
    '18 a 20 anos': '18 a 24 anos',
    '21 a 24 anos': '18 a 24 anos',
    '25 a 34 anos': '25 a 34 anos',
    '35 a 44 anos': '35 a 44 anos',
    '45 a 59 anos': '45 a 54 anos', # 45 a 54 = 2/3
    '60 a 69 anos': '55 a 65 anos', # 1/2 this + 1/3 prev 
    '70 a 79 anos': 'Superior a 65 anos',
    'Superior a 79 anos': 'Superior a 65 anos'
}
age_voters["DS_FAIXA_ETARIA2"] = age_voters['DS_FAIXA_ETARIA'].map(age_category_mapping)
print(age_voters)
age_voters.to_csv("age_voters.csv")
