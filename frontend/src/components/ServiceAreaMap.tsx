import {
  useEffect,
  useState,
} from "react";

import L from "leaflet";

import booleanPointInPolygon
  from "@turf/boolean-point-in-polygon";

import {
  CircleMarker,
  GeoJSON,
  MapContainer,
  TileLayer,
  useMap,
  useMapEvents,
} from "react-leaflet";

import type {
  GeoJsonObject,
} from "geojson";

import {
  getSaricamServiceArea,
} from "../api/serviceAreas";

import type {
  SelectedLocation,
} from "../types/location";

import type {
  ServiceAreaFeature,
} from "../types/serviceArea";


interface FocusLocation {
  latitude: number;
  longitude: number;
}


interface ServiceAreaMapProps {
  selectedLocation:
    SelectedLocation | null;

  onLocationChange: (
    location: SelectedLocation,
  ) => void;

  focusLocation:
    FocusLocation | null;
}


function FitServiceAreaBounds({
  serviceArea,
}: {
  serviceArea: ServiceAreaFeature;
}) {
  const map = useMap();

  useEffect(() => {
    const layer = L.geoJSON(
      serviceArea as GeoJsonObject,
    );

    const bounds =
      layer.getBounds();

    if (bounds.isValid()) {
      map.fitBounds(
        bounds,
        {
          padding: [24, 24],
          maxZoom: 12,
        },
      );
    }
  }, [
    map,
    serviceArea,
  ]);

  return null;
}


function NeighborhoodFocusController({
  focusLocation,
}: {
  focusLocation: FocusLocation | null;
}) {
  const map = useMap();

  useEffect(() => {
    if (!focusLocation) {
      return;
    }

    map.flyTo(
      [
        focusLocation.latitude,
        focusLocation.longitude,
      ],
      15,
      {
        animate: true,
        duration: 0.8,
      },
    );
  }, [
    focusLocation,
    map,
  ]);

  return null;
}


function MapClickHandler({
  serviceArea,
  onLocationChange,
  onInvalidSelection,
}: {
  serviceArea: ServiceAreaFeature;

  onLocationChange: (
    location: SelectedLocation,
  ) => void;

  onInvalidSelection: () => void;
}) {
  useMapEvents({
    click(event) {
      const latitude =
        event.latlng.lat;

      const longitude =
        event.latlng.lng;

      const inside =
        booleanPointInPolygon(
          [
            longitude,
            latitude,
          ],
          serviceArea,
        );

      if (!inside) {
        onInvalidSelection();
        return;
      }

      onLocationChange({
        latitude,
        longitude,
      });
    },
  });

  return null;
}


function ServiceAreaMap({
  selectedLocation,
  onLocationChange,
  focusLocation,
}: ServiceAreaMapProps) {
  const [
    serviceArea,
    setServiceArea,
  ] = useState<ServiceAreaFeature | null>(
    null,
  );

  const [
    error,
    setError,
  ] = useState<string | null>(
    null,
  );

  const [
    selectionError,
    setSelectionError,
  ] = useState<string | null>(
    null,
  );


  useEffect(() => {
    let active = true;

    async function loadServiceArea() {
      try {
        const data =
          await getSaricamServiceArea();

        if (active) {
          setServiceArea(data);
          setError(null);
        }
      } catch (err) {
        if (!active) {
          return;
        }

        const message =
          err instanceof Error
            ? err.message
            : "Hizmet alanı yüklenemedi.";

        setError(message);

        console.error(
          "Service area API error:",
          err,
        );
      }
    }

    loadServiceArea();

    return () => {
      active = false;
    };
  }, []);


  function handleValidSelection(
    location: SelectedLocation,
  ) {
    setSelectionError(null);

    onLocationChange(
      location,
    );
  }


  function handleInvalidSelection() {
    setSelectionError(
      "Seçtiğiniz konum Sarıçam hizmet "
      + "alanının dışında. Lütfen mavi "
      + "sınır içerisinden bir nokta seçin.",
    );
  }


  if (error) {
    return (
      <div className="map-error">
        <strong>
          Harita hizmet alanı yüklenemedi.
        </strong>

        <span>
          {error}
        </span>
      </div>
    );
  }


  if (!serviceArea) {
    return (
      <div className="map-loading">
        Sarıçam hizmet alanı yükleniyor...
      </div>
    );
  }


  return (
    <section className="service-area-section">
      <div className="map-heading">
        <div>
          <span className="map-kicker">
            CBS Hizmet Alanı
          </span>

          <h2>
            Başvuru Konumu
          </h2>
        </div>

        <span className="map-area-badge">
          {serviceArea.properties.name}
        </span>
      </div>


      <p className="map-description">
        Şikâyetin bulunduğu noktayı
        harita üzerinden seçin.
        Yalnızca Sarıçam hizmet alanı
        içerisindeki konumlar kabul edilir.
      </p>


      <div className="map-frame">
        <MapContainer
          center={[
            37.1110882,
            35.473158069576726,
          ]}
          zoom={10}
          scrollWheelZoom
          className="service-area-map"
        >
          <TileLayer
            attribution={
              '&copy; OpenStreetMap contributors'
            }
            url={
              "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            }
          />

          <GeoJSON
            data={
              serviceArea as GeoJsonObject
            }
            style={{
              color: "#315fd6",
              weight: 3,
              fillColor: "#4d78e8",
              fillOpacity: 0.12,
            }}
          />

          <FitServiceAreaBounds
            serviceArea={
              serviceArea
            }
          />

          <NeighborhoodFocusController
            focusLocation={
              focusLocation
            }
          />

          <MapClickHandler
            serviceArea={
              serviceArea
            }
            onLocationChange={
              handleValidSelection
            }
            onInvalidSelection={
              handleInvalidSelection
            }
          />

          {selectedLocation && (
            <CircleMarker
              center={[
                selectedLocation.latitude,
                selectedLocation.longitude,
              ]}
              radius={9}
              pathOptions={{
                color: "#173fbd",
                fillColor: "#315fd6",
                fillOpacity: 1,
                weight: 3,
              }}
            />
          )}
        </MapContainer>
      </div>


      {selectionError && (
        <div className="location-message error">
          {selectionError}
        </div>
      )}


      {selectedLocation && (
        <div className="location-message success">
          <strong>
            Konum seçildi
          </strong>

          <span>
            Başvuru konumu harita üzerinden
            kaydedildi.
          </span>
        </div>
      )}


      <div className="map-source">
        Sınır kaynağı:{" "}
        {serviceArea.properties.source}
      </div>
    </section>
  );
}


export default ServiceAreaMap;