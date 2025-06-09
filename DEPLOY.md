# DEPLOY.md

## Despliegue seguro en producción

### 1. Variables de entorno necesarias
- `METACULUS_TOKEN`: Token de API de Metaculus
- `OPENAI_API_KEY`: (opcional, si usas OpenAI)
- `ANTHROPIC_API_KEY`: (opcional, si usas Anthropic)
- `DISCORD_WEBHOOK_URL`: (opcional, para alertas/plugins)

### 2. Cómo configurar
- Crea un archivo `.env` en la raíz del proyecto (no lo subas a git)
- Usa `.env.example` como plantilla
- En servidores/CI, exporta las variables directamente en el entorno

### 3. Instalación de dependencias
```zsh
poetry install
```

### 4. Ejecución segura
```zsh
PYTHONPATH=$(pwd) poetry run python main_agent.py --mode batch --limit 3 --show-trace --dryrun
```

### 5. Validación
- El bot validará la presencia de las claves críticas al arrancar
- Si falta alguna, abortará con un mensaje claro

### 6. Seguridad
- Nunca subas `.env` ni claves a git
- Revisa los logs para evitar fugas de secretos

### 7. Actualización de claves
- Cambia las claves en el entorno o `.env` y reinicia el bot

---
Para dudas o problemas de despliegue, revisa README.md y este archivo.
