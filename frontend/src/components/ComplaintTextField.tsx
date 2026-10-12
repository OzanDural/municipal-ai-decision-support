interface ComplaintTextFieldProps {
  value: string;

  onChange: (
    value: string,
  ) => void;

  disabled?: boolean;
}


function ComplaintTextField({
  value,
  onChange,
  disabled = false,
}: ComplaintTextFieldProps) {
  return (
    <section className="complaint-section">
      <div className="address-heading">
        <span className="map-kicker">
          Başvuru Bilgileri
        </span>

        <h2>
          Şikâyet veya Talebiniz
        </h2>
      </div>

      <p className="address-description">
        Belediye hizmetleriyle ilgili
        şikâyet veya talebinizi açık ve
        anlaşılır şekilde yazın.
      </p>

      <label
        className={
          "form-field "
          + "form-field-full"
        }
      >
        <span>
          Şikâyet / Talep *
        </span>

        <textarea
          value={value}
          disabled={disabled}
          minLength={5}
          maxLength={3000}
          rows={6}
          placeholder={
            "Örn. Mahallemizdeki çöp "
            + "konteyneri birkaç gündür "
            + "boşaltılmıyor."
          }
          onChange={(event) => {
            onChange(
              event.target.value,
            );
          }}
        />
      </label>

      <div className="complaint-counter">
        {value.length} / 3000
      </div>
    </section>
  );
}


export default ComplaintTextField;