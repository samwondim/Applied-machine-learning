import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';

export const getData = async () => {
  try {
    const response = await axios.get(`${API_URL}/data`);
    return response.data;
  } catch (error) {
    console.error('Error fetching data:', error);
    throw error;
  }
};

export const postData = async (data) => {
  try {
    const response = await axios.post(`${API_URL}/data`, data);
    return response.data;
  } catch (error) {
    console.error('Error posting data:', error);
    throw error;
  }
};
