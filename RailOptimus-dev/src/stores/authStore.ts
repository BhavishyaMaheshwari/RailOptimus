
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import axios from 'axios';

// --- 1. Define Data Shapes to Match Backend ---

// This interface matches the User object sent from your backend
export interface User {
  _id: string; // MongoDB uses _id
  name: string;
  section: string;
  username: string;
  role: 'controller' | 'admin';
}

// This interface defines the data needed for an admin to create a new user
export interface NewUserData {
  name: string;
  username: string;
  password:  string;
  passwordConfirm: string;
  section: string;
  role?: 'controller' | 'admin';
}

// This interface defines the complete state of our auth store
interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  login: (username: string, password: string) => Promise<boolean>;
  signup: (userData: NewUserData) => Promise<{ success: boolean; message: string }>;
  logout: () => void;
}

// --- 2. Configure API URL ---

// The base URL for your backend user routes (configured via Vite env)
const API_BASE = (import.meta as any).env?.VITE_API_BASE || 'http://localhost:5005';
const API_URL = `${API_BASE}/api/users`;

// --- 3. Create the Zustand Store ---

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      // --- Initial State ---
      user: null,
      token: null,
      isAuthenticated: false,

      /**
       * Logs in a user by calling the backend API.
       */
      login: async (username, password) => {
        try {
          const response = await axios.post(`${API_URL}/login`, { username, password });
          
          if (response.data.status === 'success') {
            const { user, token } = response.data.data;
            set({ user, token, isAuthenticated: true });
            return true;
          }
          return false;
        } catch (error: any) {
          console.error("Login failed:", error.response?.data?.message || error.message);
          return false;
        }
      },

      /**
       * Creates a new user account. (Admin Only)
       * This function requires a valid admin token to be present in the state.
       */
      signup: async (userData: NewUserData) => {
        // const { token } = get(); // Get the current admin's token from the store

        // if (!token) {
        //   return { success: false, message: 'Authorization error: Admin not logged in.' };
        // }

        try {
          // The key is to send the admin's token in the 'Authorization' header.
          // The backend middleware will use this to verify the request.
          console.log('Sending data to:','with data:', userData)
          await axios.post(`${API_URL}/signup`, userData)
          // ({
          //   headers: {
          //     Authorization: `Bearer ${token}`,
          //   },
          // });
          
          return { success: true, message: 'User account created successfully!' };
        } catch (error: any) {
          const errorMessage = error.response?.data?.message || 'Failed to create user account.';
          console.error("Signup failed:", errorMessage);
          return { success: false, message: errorMessage };
        }
      },

      /**
       * Logs the user out by clearing their data from the state.
       */
      logout: () => {
        set({ user: null, token: null, isAuthenticated: false });
      },
    }),
    {
      // --- 4. Configure Persistence ---
      name: 'auth-storage', // The key for the data in localStorage
      // This function specifies which parts of the state to save
      partialize: (state) => ({ 
        user: state.user, 
        token: state.token,
        isAuthenticated: state.isAuthenticated 
      }),
    }
  )
);