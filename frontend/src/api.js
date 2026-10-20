import axios from 'axios';
export const api=axios.create({baseURL:import.meta.env.VITE_API_BASE_URL||'',timeout:240000});
export const auth=(session)=>({headers:{Authorization:`Bearer ${session.session_token}`}});
export const reviewAuth=(key)=>({headers:{'X-Reviewer-Key':key}});
export function errorMessage(e){return e?.response?.data?.error?.message || (e?.response?.status===422?'Check the required form fields.':e?.message) || 'Request failed.';}
export async function upload(session,file,options){const form=new FormData();form.append('image',file);Object.entries(options).forEach(([k,v])=>form.append(k,v));return (await api.post(`/api/cases/${session.id}/documents`,form,auth(session))).data;}
