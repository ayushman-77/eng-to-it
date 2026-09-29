# Build Stage
FROM node:22-alpine as build
WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .
# You can pass VITE_API_URL at build time if needed
# ARG VITE_API_URL
# ENV VITE_API_URL=$VITE_API_URL

RUN npm run build

# Production Stage
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
