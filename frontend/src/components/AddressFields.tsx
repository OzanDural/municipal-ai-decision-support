import {
  useEffect,
  useMemo,
  useState,
} from "react";

import {
  getNeighborhoods,
  getStreets,
} from "../api/locations";


export interface AddressFieldsValue {
  neighborhood: string;
  streetName: string | null;
  addressText: string;
}


interface AddressFieldsProps {
  value: AddressFieldsValue;

  onChange: (
    value: AddressFieldsValue,
  ) => void;
}


function normalizeForSearch(
  value: string,
): string {
  return (
    value
      .toLocaleLowerCase("tr-TR")
      .replaceAll("ç", "c")
      .replaceAll("ğ", "g")
      .replaceAll("ı", "i")
      .replaceAll("ö", "o")
      .replaceAll("ş", "s")
      .replaceAll("ü", "u")
      .replace(/\s+/g, " ")
      .trim()
  );
}


function normalizeManualStreet(
  value: string,
): string | null {
  const cleaned = (
    value
      .replace(/\s+/g, " ")
      .trim()
  );

  if (!cleaned) {
    return null;
  }

  return cleaned;
}


function AddressFields({
  value,
  onChange,
}: AddressFieldsProps) {
  const [
    neighborhoods,
    setNeighborhoods,
  ] = useState<string[]>([]);

  const [
    streets,
    setStreets,
  ] = useState<string[]>([]);

  const [
    streetInput,
    setStreetInput,
  ] = useState(
    value.streetName ?? "",
  );

  const [
    selectedSuggestedStreet,
    setSelectedSuggestedStreet,
  ] = useState<string | null>(
    null,
  );

  const [
    streetSuggestionsOpen,
    setStreetSuggestionsOpen,
  ] = useState(false);

  const [
    loading,
    setLoading,
  ] = useState(true);

  const [
    error,
    setError,
  ] = useState<string | null>(
    null,
  );


  useEffect(() => {
    let active = true;

    async function loadLocationData() {
      try {
        const [
          neighborhoodData,
          streetData,
        ] = await Promise.all([
          getNeighborhoods(),
          getStreets(),
        ]);

        if (!active) {
          return;
        }

        setNeighborhoods(
          neighborhoodData.neighborhoods,
        );

        setStreets(
          streetData.streets,
        );

        setError(null);
      } catch (err) {
        if (!active) {
          return;
        }

        const message =
          err instanceof Error
            ? err.message
            : (
              "Adres verileri "
              + "yüklenemedi."
            );

        setError(message);
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    loadLocationData();

    return () => {
      active = false;
    };
  }, []);


  const uniqueStreets =
    useMemo(() => {
      const byNormalizedName =
        new Map<string, string>();

      for (const street of streets) {
        const key =
          normalizeForSearch(
            street,
          );

        if (
          key
          && !byNormalizedName.has(key)
        ) {
          byNormalizedName.set(
            key,
            street,
          );
        }
      }

      return Array.from(
        byNormalizedName.values(),
      );
    }, [
      streets,
    ]);


  const filteredStreets =
    useMemo(() => {
      const query =
        normalizeForSearch(
          streetInput,
        );

      if (!query) {
        return [];
      }

      const startsWithMatches:
        string[] = [];

      const containsMatches:
        string[] = [];

      for (
        const street
        of uniqueStreets
      ) {
        const normalizedStreet =
          normalizeForSearch(
            street,
          );

        if (
          normalizedStreet
            .startsWith(query)
        ) {
          startsWithMatches.push(
            street,
          );

          continue;
        }

        if (
          normalizedStreet
            .includes(query)
        ) {
          containsMatches.push(
            street,
          );
        }
      }

      return [
        ...startsWithMatches,
        ...containsMatches,
      ].slice(
        0,
        8,
      );
    }, [
      streetInput,
      uniqueStreets,
    ]);


  function handleNeighborhoodChange(
    neighborhood: string,
  ) {
    onChange({
      ...value,
      neighborhood,
    });
  }


  function handleStreetInputChange(
    input: string,
  ) {
    setStreetInput(
      input,
    );

    setSelectedSuggestedStreet(
      null,
    );

    setStreetSuggestionsOpen(
      true,
    );

    onChange({
      ...value,
      streetName:
        normalizeManualStreet(
          input,
        ),
    });
  }


  function handleStreetSelect(
    street: string,
  ) {
    setStreetInput(
      street,
    );

    setSelectedSuggestedStreet(
      street,
    );

    setStreetSuggestionsOpen(
      false,
    );

    onChange({
      ...value,
      streetName: street,
    });
  }


  function handleStreetBlur() {
    window.setTimeout(
      () => {
        setStreetSuggestionsOpen(
          false,
        );

        const normalized =
          normalizeManualStreet(
            streetInput,
          );

        if (
          normalized !== null
          && normalized
          !== streetInput
        ) {
          setStreetInput(
            normalized,
          );
        }

        onChange({
          ...value,
          streetName:
            normalized,
        });
      },
      120,
    );
  }


  function handleAddressTextChange(
    addressText: string,
  ) {
    onChange({
      ...value,
      addressText,
    });
  }


  return (
    <section className="address-section">
      <div className="address-heading">
        <span className="map-kicker">
          Adres Bilgileri
        </span>

        <h2>
          Başvuru Adresi
        </h2>
      </div>

      <p className="address-description">
        Mahallenizi seçin ve varsa
        sokak, cadde, bulvar veya yol
        bilgisini belirtin.
      </p>

      {error && (
        <div className="location-message error">
          {error}
        </div>
      )}

      <div className="address-grid">
        <label className="form-field">
          <span>
            Mahalle *
          </span>

          <select
            value={
              value.neighborhood
            }
            disabled={
              loading
              || Boolean(error)
            }
            onChange={(event) => {
              handleNeighborhoodChange(
                event.target.value,
              );
            }}
          >
            <option value="">
              {loading
                ? "Mahalleler yükleniyor..."
                : "Mahalle seçiniz"}
            </option>

            {neighborhoods.map(
              (neighborhood) => (
                <option
                  key={neighborhood}
                  value={neighborhood}
                >
                  {neighborhood}
                </option>
              ),
            )}
          </select>
        </label>

        <div className="form-field">
          <label
            htmlFor="street-name"
          >
            Sokak / Cadde / Bulvar / Yol
          </label>

          <div className="autocomplete">
            <input
              id="street-name"
              type="text"
              autoComplete="off"
              maxLength={160}
              placeholder={
                loading
                  ? "Yollar yükleniyor..."
                  : "Yol adı yazmaya başlayın"
              }
              disabled={
                loading
                || Boolean(error)
              }
              value={
                streetInput
              }
              onFocus={() => {
                setStreetSuggestionsOpen(
                  true,
                );
              }}
              onBlur={
                handleStreetBlur
              }
              onChange={(event) => {
                handleStreetInputChange(
                  event.target.value,
                );
              }}
            />

            {(
              streetSuggestionsOpen
              && streetInput.trim()
              && filteredStreets.length > 0
            ) && (
              <div
                className="autocomplete-menu"
              >
                {filteredStreets.map(
                  (street) => (
                    <button
                      key={
                        normalizeForSearch(
                          street,
                        )
                      }
                      type="button"
                      className={
                        "autocomplete-option"
                      }
                      onMouseDown={(
                        event
                      ) => {
                        event.preventDefault();

                        handleStreetSelect(
                          street,
                        );
                      }}
                    >
                      {street}
                    </button>
                  ),
                )}
              </div>
            )}
          </div>

          {(
            streetInput.trim()
            && selectedSuggestedStreet
          ) && (
            <small
              className={
                "field-helper success"
              }
            >
              Öneriden seçildi:{" "}
              {selectedSuggestedStreet}
            </small>
          )}

          {(
            streetInput.trim()
            && !selectedSuggestedStreet
          ) && (
            <small className="field-helper">
              Listeden bir yol seçebilir
              veya listede bulunmuyorsa
              yol adını manuel olarak
              yazabilirsiniz.
            </small>
          )}

          {!streetInput.trim() && (
            <small className="field-helper">
              Bu alan isteğe bağlıdır.
            </small>
          )}
        </div>

        <label
          className={
            "form-field "
            + "form-field-full"
          }
        >
          <span>
            Adres Tarifi
          </span>

          <textarea
            value={
              value.addressText
            }
            maxLength={500}
            rows={3}
            placeholder={
              "Örn. marketin karşısı, "
              + "konteynerin yanı"
            }
            onChange={(event) => {
              handleAddressTextChange(
                event.target.value,
              );
            }}
          />
        </label>
      </div>
    </section>
  );
}


export default AddressFields;