FROM mcr.microsoft.com/playwright:v1.45.0-jammy

WORKDIR /app

COPY package.json .
RUN npm install --omit=dev

COPY server.js .

ENV PORT=7003

EXPOSE 7003

CMD ["node", "server.js"]
