import {
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import AddressFields from
  "../components/AddressFields";

import type {
  AddressFieldsValue,
} from "../components/AddressFields";

import ComplaintTextField from
  "../components/ComplaintTextField";

import ServiceAreaMap from
  "../components/ServiceAreaMap";

import {
  createComplaint,
} from "../api/complaints";

import {
  getNeighborhoodCenters,
} from "../api/locations";

import type {
  ComplaintCreateResponse,
} from "../types/complaint";

import type {
  NeighborhoodCenter,
} from "../types/locationData";

import type {
  SelectedLocation,
} from "../types/location";
import {
  useNavigate,
} from "react-router-dom";

function PublicPage() {
    const navigate =
    useNavigate();
  const [
    complaintText,
    setComplaintText,
  ] = useState("");

  const [
    selectedLocation,
    setSelectedLocation,
  ] = useState<SelectedLocation | null>(
    null,
  );

  const [
    addressFields,
    setAddressFields,
  ] = useState<AddressFieldsValue>({
    neighborhood: "",
    streetName: null,
    addressText: "",
  });

  const [
    neighborhoodCenters,
    setNeighborhoodCenters,
  ] = useState<NeighborhoodCenter[]>(
    [],
  );

  const [
    submitting,
    setSubmitting,
  ] = useState(false);

  const [
    submitError,
    setSubmitError,
  ] = useState<string | null>(
    null,
  );

  const [
    submitResult,
    setSubmitResult,
  ] = useState<ComplaintCreateResponse | null>(
    null,
  );

  const [
    addressFormKey,
    setAddressFormKey,
  ] = useState(0);

  const submitLockRef =
    useRef(false);


  useEffect(() => {
    let active = true;

    async function loadNeighborhoodCenters() {
      try {
        const data =
          await getNeighborhoodCenters();

        if (!active) {
          return;
        }

        setNeighborhoodCenters(
          data.centers,
        );
      } catch (error) {
        console.error(
          (
            "Neighborhood centers "
            + "could not be loaded."
          ),
          error,
        );
      }
    }

    loadNeighborhoodCenters();

    return () => {
      active = false;
    };
  }, []);


  const selectedNeighborhoodCenter =
    useMemo(() => {
      if (
        !addressFields.neighborhood
      ) {
        return null;
      }

      return (
        neighborhoodCenters.find(
          (center) =>
            center.name
            === addressFields.neighborhood,
        )
        ?? null
      );
    }, [
      addressFields.neighborhood,
      neighborhoodCenters,
    ]);


  const canSubmit =
    complaintText.trim().length >= 5
    && Boolean(
      addressFields.neighborhood,
    )
    && selectedLocation !== null
    && !submitting
    && submitResult === null;


  async function handleSubmit() {
    if (
      submitLockRef.current
      || submitting
      || submitResult !== null
    ) {
      return;
    }

    setSubmitError(
      null,
    );

    const cleanedComplaintText =
      complaintText
        .replace(/\s+/g, " ")
        .trim();

    if (
      cleanedComplaintText.length < 5
    ) {
      setSubmitError(
        "Lütfen şikâyet veya talebinizi "
        + "en az 5 karakter olacak şekilde yazın.",
      );

      return;
    }

    if (
      !addressFields.neighborhood
    ) {
      setSubmitError(
        "Lütfen mahalle seçin.",
      );

      return;
    }

    if (!selectedLocation) {
      setSubmitError(
        "Lütfen başvuru konumunu "
        + "harita üzerinden seçin.",
      );

      return;
    }

    submitLockRef.current = true;

    try {
      setSubmitting(
        true,
      );

      const result =
        await createComplaint({
          complaint_text:
            cleanedComplaintText,

          neighborhood:
            addressFields.neighborhood,

          street_name:
            addressFields.streetName,

          address_text:
            addressFields.addressText.trim()
              ? addressFields.addressText
                  .replace(/\s+/g, " ")
                  .trim()
              : null,

          latitude:
            selectedLocation.latitude,

          longitude:
            selectedLocation.longitude,
        });

      setSubmitResult(
        result,
      );

      setSubmitError(
        null,
      );
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : "Başvuru gönderilemedi.";

      setSubmitError(
        message,
      );
    } finally {
      submitLockRef.current = false;

      setSubmitting(
        false,
      );
    }
  }


  function handleNewApplication() {
    submitLockRef.current = false;

    setComplaintText(
      "",
    );

    setAddressFields({
      neighborhood: "",
      streetName: null,
      addressText: "",
    });

    setSelectedLocation(
      null,
    );

    setSubmitError(
      null,
    );

    setSubmitResult(
      null,
    );

    setSubmitting(
      false,
    );

    setAddressFormKey(
      (current) =>
        current + 1,
    );
  }


  return (
    <main className="page public-page">
      <div className="public-container">
        <section className="hero-card public-hero">
          <button
            type="button"
            className="public-staff-login"
            onClick={() => {
              navigate("/login");
            }}
          >
            Personel Girişi
          </button>
          <div className="eyebrow">
            Sarıçam Belediyesi
          </div>

          <h1>
            Akıllı Belediye
            <br />
            Başvuru Sistemi
          </h1>

          <p>
            Belediye hizmetleriyle ilgili
            talebinizi iletin ve konumunuzu
            harita üzerinden belirleyin.
          </p>

          <div className="status-pill">
            Vatandaş Başvuru Alanı
          </div>

          <div className="public-system-status">
            <span
              className="public-system-status-dot"
              aria-hidden="true"
            />

          <div>
            <strong>
              Başvuru sistemi kullanıma hazır
            </strong>

            <span>
              Başvurunuzu aşağıdaki formdan
              oluşturabilirsiniz.
            </span>
          </div>
        </div>
        </section>

        <ComplaintTextField
          value={
            complaintText
          }
          onChange={
            setComplaintText
          }
          disabled={
            submitting
            || submitResult !== null
          }
        />

        <AddressFields
          key={
            addressFormKey
          }
          value={
            addressFields
          }
          onChange={
            setAddressFields
          }
        />

        <ServiceAreaMap
          selectedLocation={
            selectedLocation
          }
          onLocationChange={
            setSelectedLocation
          }
          focusLocation={
            selectedNeighborhoodCenter
          }
        />

        <section className="submit-section">
          {submitError && (
            <div className="location-message error">
              {submitError}
            </div>
          )}

          {submitResult && (
            <div className="application-success">
              <strong>
                Başvurunuz alındı.
              </strong>

              <span>
                Başvuru Numaranız
              </span>

              <div className="application-number">
                {submitResult.application_no}
              </div>

              <p>
                Başvuru numaranızı
                saklayabilirsiniz.
              </p>
            </div>
          )}

          {submitResult ? (
            <button
              type="button"
              className="submit-complaint-button"
              onClick={
                handleNewApplication
              }
            >
              Yeni Başvuru Oluştur
            </button>
          ) : (
            <button
              type="button"
              className="submit-complaint-button"
              disabled={
                !canSubmit
              }
              onClick={
                handleSubmit
              }
            >
              {submitting
                ? "Başvuru gönderiliyor..."
                : "Başvuruyu Gönder"}
            </button>
          )}

          {(
            !submitResult
            && !selectedLocation
          ) && (
            <small className="submit-helper">
              Başvuruyu göndermek için
              haritadan konum seçmeniz gerekir.
            </small>
          )}
        </section>
      </div>
    </main>
  );
}


export default PublicPage;