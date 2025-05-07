# Use Node.js base image
FROM node:18

# Create app directory
WORKDIR /app

# Copy package.json and install dependencies
COPY package*.json ./
RUN npm install

# Copy the rest of the app
COPY . .

# Expose port (must match the one used in your app)
EXPOSE 3000

# Start the server
CMD ["npm", "start"]
