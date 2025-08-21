FROM  jenkins/jenkins:lts-jdk17

# Switch to root user to install dependencies

USER root

# Install Docker CLI
RUN apt-get update && \
apt-get install -y docker.io && \
rm -rf /var/lib/apt/lists/*

# Add Jenkins user to docker group (so it can access /var/run/docker.sock)
RUN groupadd -f docker && usermod -aG docker jenkins

USER jenkins