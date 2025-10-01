# Multi-stage Docker build for EEG Stress Detection System
# Stage 1: Build Next.js frontend
FROM node:18-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy package files
COPY package.json package-lock.json* ./

# Install dependencies
RUN npm ci --only=production

# Copy frontend source
COPY app ./app
COPY components ./components
COPY lib ./lib
COPY contexts ./contexts
COPY hooks ./hooks
COPY types ./types
COPY utils ./utils
COPY next.config.mjs next-env.d.ts tailwind.config.ts postcss.config.mjs tsconfig.json ./
COPY components.json ./

# Build frontend
RUN npm run build

# Stage 2: Python backend
FROM python:3.9-slim AS backend

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy Python requirements
COPY requirements-api.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy Python backend
COPY master_eeg_analyzer.py ./
COPY emotion_model_new.pth ./

# Stage 3: Final runtime image
FROM python:3.9-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python backend from backend stage
COPY --from=backend /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=backend /usr/local/bin /usr/local/bin
COPY --from=backend /app/master_eeg_analyzer.py ./
COPY --from=backend /app/emotion_model_new.pth ./
COPY requirements-api.txt ./

# Copy frontend build from frontend stage
COPY --from=frontend-builder /app/frontend/.next ./.next
COPY --from=frontend-builder /app/frontend/public ./public
COPY --from=frontend-builder /app/frontend/package.json ./package.json
COPY --from=frontend-builder /app/frontend/node_modules ./node_modules

# Copy additional files
COPY eeg_api_server.py ./
COPY next.config.mjs ./

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose ports
EXPOSE 3000 8001 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000/api/health || exit 1

# Default command - start both services
CMD ["sh", "-c", "python eeg_api_server.py --port 8001 --mode realtime & npm start"]
