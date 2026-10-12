import {
  Navigate,
  Route,
  Routes,
} from "react-router-dom";

import RequireRole from
  "./components/RequireRole";

import LoginPage from
  "./pages/LoginPage";

import ManagerPage from
  "./pages/ManagerPage";

import OperatorPage from
  "./pages/OperatorPage";

import PublicPage from
  "./pages/PublicPage";

import "./App.css";


function App() {
  return (
    <Routes>
      <Route
        path="/"
        element={
          <Navigate
            to="/public"
            replace
          />
        }
      />

      <Route
        path="/public"
        element={
          <PublicPage />
        }
      />

      <Route
        path="/login"
        element={
          <LoginPage />
        }
      />

      <Route
        path="/operator"
        element={
          <RequireRole
            role="OPERATOR"
          >
            <OperatorPage />
          </RequireRole>
        }
      />

      <Route
        path="/manager"
        element={
          <RequireRole
            role="MANAGER"
          >
            <ManagerPage />
          </RequireRole>
        }
      />

      <Route
        path="*"
        element={
          <Navigate
            to="/public"
            replace
          />
        }
      />
    </Routes>
  );
}


export default App;