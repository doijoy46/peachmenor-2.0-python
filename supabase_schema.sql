-- Peachmenor Supabase Schema
-- Run this in the Supabase SQL Editor

create table public.jobs (
  id uuid primary key default gen_random_uuid(),
  filename text not null,
  rendered_url text,
  scene_analysis jsonb,
  created_at timestamptz not null default now()
);

create table public.crops (
  id uuid primary key default gen_random_uuid(),
  job_id uuid not null references public.jobs(id) on delete cascade,
  label text,
  confidence real,
  crop_url text,
  generated_url text,
  generated_error text,
  metadata jsonb,
  created_at timestamptz not null default now()
);

create index idx_crops_job_id on public.crops(job_id);
