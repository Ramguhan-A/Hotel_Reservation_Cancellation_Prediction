pipeline {
    agent any

    stages {
        stage('Cloning Github repo to Jenkins') {
            steps {
                script {
                    echo 'Cloning Github repo to jenkins...'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/Ramguhan-A/Hotel_Reservation_Cancellation_Prediction.git']])
                }
            }
        }

        stage("Build Docker Image") {
            steps {
                echo 'Building Docker image...'
                sh 'docker build -t hbr_mlops .'
            }
        }
    }
}
