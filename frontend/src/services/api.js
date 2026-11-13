/**
 * API Service Client for TerraSpectra Backend.
 */

import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const apiService = {
  async getStatus() {
    const response = await apiClient.get('/status');
    return response.data;
  },

  async getSpectralSignatures() {
    const response = await apiClient.get('/spectral/signatures');
    return response.data;
  },

  async getWavelengths() {
    const response = await apiClient.get('/spectral/wavelengths');
    return response.data;
  },

  async getFarmGeoJSON() {
    const response = await apiClient.get('/farm/geojson');
    return response.data;
  },

  async getPCASummary() {
    const response = await apiClient.get('/pipeline/pca-summary');
    return response.data;
  },

  async triggerSimulation(seed = null, foci = 3) {
    const response = await apiClient.post('/pipeline/simulate', null, {
      params: { seed, foci },
    });
    return response.data;
  },

  // 3D-CNN Model API
  async getModelArchitecture() {
    const response = await apiClient.get('/model/architecture');
    return response.data;
  },

  async trainStep(epochs = 1) {
    const response = await apiClient.post('/model/train-step', null, {
      params: { epochs },
    });
    return response.data;
  },

  async predictSceneRisk() {
    const response = await apiClient.post('/model/predict-scene');
    return response.data;
  },

  // Mid-Project Review Audit & Deck.gl Heatmap API (Day 5)
  async getFarmHeatmap() {
    const response = await apiClient.get('/farm/heatmap');
    return response.data;
  },

  async getMidProjectReviewAudit() {
    const response = await apiClient.get('/audit/mid-project-review');
    return response.data;
  },
};

export default apiService;
