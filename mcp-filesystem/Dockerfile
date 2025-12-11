FROM node:22-alpine

WORKDIR /app

COPY package.json ./
RUN npm install

COPY index.js .

ENV PORT=7001

EXPOSE 7001

CMD ["node", "index.js"]
