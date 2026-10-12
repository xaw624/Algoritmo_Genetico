#!/usr/bin/env bash
# Instala el simulador como servicio systemd (streamlit@genetico) en un
# servidor dedicado con Ubuntu Server 24.04 LTS (p. ej. AWS EC2 t4g.micro).
#
# Uso, como el usuario ubuntu (necesita sudo):
#   git clone https://github.com/xaw624/Algoritmo_Genetico.git ~/Algoritmo_Genetico
#   bash ~/Algoritmo_Genetico/deploy/instalar.sh
#
# Para conectar además el Cloudflare Tunnel, pasa el token del túnel
# (Cloudflare Zero Trust -> Networks -> Tunnels -> tu túnel):
#   CLOUDFLARE_TUNNEL_TOKEN='<token>' bash ~/Algoritmo_Genetico/deploy/instalar.sh
#
# Se puede volver a ejecutar sin problema: actualiza dependencias y reinicia.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY="$REPO/deploy"
INSTANCIA="genetico"
PUERTO="$(sed -n 's/^PORT=//p' "$DEPLOY/genetico.env")"

paso() { printf '\n==> %s\n' "$*"; }
error() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

# -----------------------------------------------------------
# Comprobaciones
# Las rutas de deploy/ suponen el usuario ubuntu y el repo en su home.
# -----------------------------------------------------------
paso "Comprobando el entorno"

[[ "$(id -un)" == "ubuntu" ]] \
    || error "Ejecuta el script como el usuario 'ubuntu' (no con sudo)."
[[ "$REPO" == "/home/ubuntu/Algoritmo_Genetico" ]] \
    || error "El repo debe estar en /home/ubuntu/Algoritmo_Genetico (está en $REPO)."
command -v systemctl >/dev/null || error "Se necesita systemd."
sudo -v || error "Se necesita sudo."

# En un servidor compartido la plantilla puede existir con otra configuración
# (otro WorkingDirectory): reemplazarla afectaría a las demás apps.
PLANTILLA=/etc/systemd/system/streamlit@.service
if [[ -f "$PLANTILLA" ]] && ! cmp -s "$PLANTILLA" "$DEPLOY/streamlit@.service"; then
    error "Ya existe $PLANTILLA y es distinta a deploy/streamlit@.service.
       Este script es para un servidor dedicado. En un servidor compartido,
       copia a mano deploy/genetico.env (ajustando PYTHON_BIN) y deploy/limites.conf."
fi

PY_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
case "$PY_VERSION" in
    3.11|3.12|3.13) echo "Python $PY_VERSION" ;;
    *) error "Python $PY_VERSION no está soportado (se requiere 3.11-3.13). Usa Ubuntu Server 24.04 LTS." ;;
esac

# -----------------------------------------------------------
# Paquetes del sistema
# -----------------------------------------------------------
paso "Instalando paquetes del sistema"
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv curl

# -----------------------------------------------------------
# Swap: en instancias de 1 GB evita que pip o un pico de memoria
# terminen en el OOM killer
# -----------------------------------------------------------
if [[ -z "$(swapon --show --noheadings)" ]] \
    && (( $(awk '/MemTotal/ {print $2}' /proc/meminfo) < 2 * 1024 * 1024 )); then
    paso "Creando swap de 1 GB"
    sudo fallocate -l 1G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile >/dev/null
    sudo swapon /swapfile
    grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

# -----------------------------------------------------------
# Entorno virtual y dependencias
# -----------------------------------------------------------
paso "Creando el entorno virtual e instalando dependencias"
[[ -d "$REPO/.venv" ]] || python3 -m venv "$REPO/.venv"
"$REPO/.venv/bin/python" -m pip install --quiet --upgrade pip
"$REPO/.venv/bin/python" -m pip install --quiet -r "$REPO/requirements.txt"

# -----------------------------------------------------------
# Servicio systemd
# -----------------------------------------------------------
paso "Instalando el servicio streamlit@$INSTANCIA"
sudo install -m 0755 -o root -g root "$DEPLOY/run_streamlit.sh" /usr/local/bin/run_streamlit.sh
sudo install -m 0644 -o root -g root "$DEPLOY/streamlit@.service" "$PLANTILLA"
sudo install -D -m 0640 -o root -g ubuntu "$DEPLOY/genetico.env" "/etc/streamlit/$INSTANCIA.env"
sudo install -D -m 0644 -o root -g root "$DEPLOY/limites.conf" \
    "/etc/systemd/system/streamlit@$INSTANCIA.service.d/limites.conf"
sudo systemctl daemon-reload
sudo systemctl enable "streamlit@$INSTANCIA" >/dev/null
sudo systemctl restart "streamlit@$INSTANCIA"

for _ in $(seq 1 30); do
    curl -sf "http://127.0.0.1:$PUERTO/_stcore/health" >/dev/null && break
    sleep 1
done
curl -sf "http://127.0.0.1:$PUERTO/_stcore/health" >/dev/null \
    || error "La app no responde en el puerto $PUERTO. Revisa: journalctl -u streamlit@$INSTANCIA -n 50"
echo "La app responde en http://127.0.0.1:$PUERTO"

# -----------------------------------------------------------
# Cloudflare Tunnel (opcional)
# -----------------------------------------------------------
if [[ -n "${CLOUDFLARE_TUNNEL_TOKEN:-}" ]]; then
    paso "Instalando cloudflared y conectando el túnel"
    if ! command -v cloudflared >/dev/null; then
        sudo mkdir -p --mode=0755 /usr/share/keyrings
        curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg \
            | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null
        echo 'deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main' \
            | sudo tee /etc/apt/sources.list.d/cloudflared.list >/dev/null
        sudo apt-get update -qq
        sudo apt-get install -y -qq cloudflared
    fi
    if systemctl is-enabled cloudflared >/dev/null 2>&1; then
        echo "cloudflared ya está instalado como servicio; no se modifica."
    else
        sudo cloudflared service install "$CLOUDFLARE_TUNNEL_TOKEN"
    fi
    systemctl is-active cloudflared >/dev/null && echo "cloudflared activo."
fi

# -----------------------------------------------------------
# Resumen
# -----------------------------------------------------------
paso "Listo"
systemctl status "streamlit@$INSTANCIA" --no-pager | head -8
cat <<EOF

Siguiente paso: en Cloudflare, publica el hostname del túnel hacia
  http://127.0.0.1:$PUERTO
(ver DOCUMENTACION.md, sección 9).

Operación:
  journalctl -u streamlit@$INSTANCIA -f                 # logs
  git -C $REPO pull && bash $DEPLOY/instalar.sh         # actualizar
EOF
