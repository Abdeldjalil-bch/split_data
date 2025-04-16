import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
import io
import zipfile

# Streamlit page configuration
st.set_page_config(page_title="Split data")

# File uploader
file = st.file_uploader(
    label="Upload Train and Test Datasets",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=False,
    help="First upload train dataset, then test dataset (optional)"
)

# Initialize dataframes
data = pd.DataFrame()
train = pd.DataFrame()
test = pd.DataFrame()

if file:
    if file.type == "text/csv":
        data = pd.read_csv(file)
    else:
        data = pd.read_excel(file)
    
    # Afficher les premières lignes du dataset
    st.write("Aperçu des données:")
    st.dataframe(data.head())
    
    # Options pour la séparation train-test
    st.title("Paramètres de séparation des données")
    
    # Option pour choisir la taille du test
    test_size = st.slider(
        "Taille du jeu de test (%)",
        min_value=10,
        max_value=50,
        value=20,
        help="Pourcentage des données à utiliser pour le test"
    ) / 100
    
    # Option pour choisir le random state
    random_state = st.number_input(
        "Random State",
        min_value=0,
        max_value=1000,
        value=42,
        help="Valeur pour garantir la reproductibilité"
    )
    
    # Option pour choisir la colonne de stratification
    strat_options = [None] + list(data.select_dtypes(include=['object', 'category', 'bool', 'int']).columns)
    strat_col = st.selectbox(
        "Colonne de stratification (optionnelle)",
        options=strat_options,
        index=0,
        help="Colonne à utiliser pour la stratification (catégorielle ou binaire de préférence)"
    )
    
    # Placeholder for download buttons
    download_placeholder = st.empty()
    
    # Bouton pour effectuer la séparation
    if st.button("Diviser les données"):
        try:
            if strat_col:
                train, test = train_test_split(
                    data,
                    test_size=test_size,
                    random_state=int(random_state),
                    stratify=data[strat_col]
                )
                st.success(f"Données divisées avec stratification sur '{strat_col}'")
            else:
                train, test = train_test_split(
                    data,
                    test_size=test_size,
                    random_state=int(random_state)
                )
                st.success("Données divisées sans stratification")
            
            # Afficher les informations sur la séparation
            st.write(f"Taille du jeu d'entraînement: {train.shape[0]} lignes")
            st.write(f"Taille du jeu de test: {test.shape[0]} lignes")
            
            # Convert DataFrames to CSV for download
            train_csv = train.to_csv(index=False)
            test_csv = test.to_csv(index=False)
            
            # Display individual download buttons
            with download_placeholder.container():
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.download_button(
                        label="Download Train File",
                        data=train_csv,
                        file_name="train.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    st.download_button(
                        label="Download Test File",
                        data=test_csv,
                        file_name="test.csv",
                        mime="text/csv"
                    )
                
                # Create a zip file containing both datasets
                with col3:
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
                        zip_file.writestr('train.csv', train_csv)
                        zip_file.writestr('test.csv', test_csv)
                    
                    # Reset buffer position to the beginning
                    zip_buffer.seek(0)
                    
                    # Download button for the zip file
                    st.download_button(
                        label="Download Both Files (ZIP)",
                        data=zip_buffer,
                        file_name="train_test_data.zip",
                        mime="application/zip"
                    )
                    
        except Exception as e:
            st.error(f"Une erreur s'est produite lors de la division des données: {e}")
