FROM node:18

WORKDIR /app

# Copy package.json from the subfolder
COPY package*.json ./

RUN npm install

# Copy the rest of the backend code
COPY backend/. .

EXPOSE 3000

CMD ["npm", "start"]
