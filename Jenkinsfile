pipeline {
    agent {
        dockerfile true
    }

    parameters {
        string(name: 'NUM_EPOCHS', defaultValue: '10', description: 'Liczba epok uczenia modelu')
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Preprocess data') {
            steps {
                sh 'python preprocess_data.py'
            }
        }

        stage('Train model') {
            steps {
                sh 'NUM_EPOCHS=${NUM_EPOCHS} python train_torch_model.py'
            }
        }

        stage('Predict on test data') {
            steps {
                sh 'python predict_torch_model.py'
            }
        }

        stage('Archive artifacts') {
            steps {
                archiveArtifacts artifacts: 'artifacts/torch_revenue_model.pth,artifacts/torch_revenue_model_features.txt,artifacts/torch_revenue_mlflow_model_uri.txt,artifacts/torch_revenue_model_registry_info.txt,artifacts/torch_revenue_model_card.md,artifacts/torch_revenue_predictions.csv,artifacts/mlflow_registry.db,artifacts/mlruns/**', fingerprint: true, onlyIfSuccessful: true
            }
        }
    }
}