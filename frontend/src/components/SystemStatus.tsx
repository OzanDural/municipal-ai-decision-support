import {
  useEffect,
  useState,
} from "react";

import {
  getHealth,
} from "../api/health";

import type {
  HealthResponse,
} from "../types/health";


function SystemStatus() {
  const [
    health,
    setHealth,
  ] = useState<HealthResponse | null>(
    null,
  );

  const [
    error,
    setError,
  ] = useState<string | null>(
    null,
  );

  useEffect(() => {
    let active = true;

    async function loadHealth() {
      try {
        const data =
          await getHealth();

        if (active) {
          setHealth(data);
          setError(null);
        }
      } catch (err) {
        if (active) {
          const message =
            err instanceof Error
              ? err.message
              : "Bilinmeyen bağlantı hatası.";

          setError(message);

          console.error(
            "Health API error:",
            err,
          );
        }
      }
    }

    loadHealth();

    return () => {
      active = false;
    };
  }, []);

  if (error) {
    return (
      <div className="system-status error">
        <strong>
          Sistem bağlantısı başarısız
        </strong>

        <span>
          {error}
        </span>
      </div>
    );
  }

  if (!health) {
    return (
      <div className="system-status">
        Sistem durumu kontrol ediliyor...
      </div>
    );
  }

  return (
    <div className="system-status">
      <div>
        <span className="status-label">
          Backend
        </span>

        <strong>
          {health.status === "ok"
            ? "Çalışıyor"
            : health.status}
        </strong>
      </div>

      <div>
        <span className="status-label">
          PostgreSQL
        </span>

        <strong>
          {health.database.connected
            ? "Bağlı"
            : "Bağlı değil"}
        </strong>
      </div>

      <div>
        <span className="status-label">
          PostGIS
        </span>

        <strong>
          {health.postgis.available
            ? "Aktif"
            : "Pasif"}
        </strong>
      </div>

      <div>
        <span className="status-label">
          BERTurk
        </span>

        <strong>
          {health.model.loaded
            ? "Yüklü"
            : "Yüklü değil"}
        </strong>
      </div>

      <div>
        <span className="status-label">
          Device
        </span>

        <strong>
          {health.model.device.toUpperCase()}
        </strong>
      </div>
    </div>
  );
}

export default SystemStatus;
