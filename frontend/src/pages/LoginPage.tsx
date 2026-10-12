import {
  useState,
} from "react";

import {
  useNavigate,
} from "react-router-dom";

import {
  login,
} from "../api/auth";


function LoginPage() {
  const navigate =
    useNavigate();

  const [
    username,
    setUsername,
  ] = useState("");

  const [
    password,
    setPassword,
  ] = useState("");

  const [
    submitting,
    setSubmitting,
  ] = useState(false);

  const [
    error,
    setError,
  ] = useState<string | null>(
    null,
  );


  async function handleSubmit(
    event:
      React.FormEvent<HTMLFormElement>,
  ) {
    event.preventDefault();

    if (submitting) {
      return;
    }

    const cleanedUsername =
      username.trim();

    if (
      cleanedUsername.length < 3
    ) {
      setError(
        "Kullanıcı adınızı kontrol edin.",
      );

      return;
    }

    if (
      password.length < 8
    ) {
      setError(
        "Şifrenizi kontrol edin.",
      );

      return;
    }

    try {
      setSubmitting(
        true,
      );

      setError(
        null,
      );

      const result =
        await login({
          username:
            cleanedUsername,

          password,
        });

      sessionStorage.setItem(
        "access_token",
        result.access_token,
      );

      sessionStorage.setItem(
        "user_role",
        result.role,
      );

      sessionStorage.setItem(
        "username",
        result.username,
      );

      if (
        result.role === "OPERATOR"
      ) {
        navigate(
          "/operator",
          {
            replace: true,
          },
        );

        return;
      }

      if (
        result.role === "MANAGER"
      ) {
        navigate(
          "/manager",
          {
            replace: true,
          },
        );

        return;
      }

      sessionStorage.clear();

      setError(
        "Bu kullanıcı rolü "
        + "desteklenmiyor.",
      );
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : "Giriş yapılamadı.";

      setError(
        message,
      );
    } finally {
      setSubmitting(
        false,
      );
    }
  }


  return (
    <main className="page login-page">
      <section className="hero-card login-card">
        <div className="eyebrow">
          Personel Girişi
        </div>

        <h1>
          Yönetim
          <br />
          Paneli
        </h1>

        <p>
          Yetkili belediye personeli
          kullanıcı adı ve şifresiyle
          sisteme giriş yapabilir.
        </p>

        <form
          className="login-form"
          onSubmit={
            handleSubmit
          }
        >
          <label className="form-field">
            <span>
              Kullanıcı Adı
            </span>

            <input
              type="text"
              autoComplete="username"
              value={
                username
              }
              disabled={
                submitting
              }
              onChange={(event) => {
                setUsername(
                  event.target.value,
                );
              }}
            />
          </label>

          <label className="form-field">
            <span>
              Şifre
            </span>

            <input
              type="password"
              autoComplete="current-password"
              value={
                password
              }
              disabled={
                submitting
              }
              onChange={(event) => {
                setPassword(
                  event.target.value,
                );
              }}
            />
          </label>

          {error && (
            <div className="location-message error">
              {error}
            </div>
          )}

          <button
            type="submit"
            className="submit-complaint-button"
            disabled={
              submitting
            }
          >
            {submitting
              ? "Giriş yapılıyor..."
              : "Giriş Yap"}
          </button>
        </form>

        <button
          type="button"
          className="login-public-link"
          onClick={() => {
            navigate(
              "/public",
            );
          }}
        >
          Vatandaş başvuru alanına dön
        </button>
      </section>
    </main>
  );
}


export default LoginPage;