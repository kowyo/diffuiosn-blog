import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";

const posts = defineCollection({
  loader: glob({ pattern: "**/*.{md,mdx}", base: "./src/content/posts" }),
  schema: z.object({
    title: z.string(),
    description: z.string().optional(),
    date: z.coerce.date(),
    authors: z
      .union([
        z.string(),
        z.array(
          z.object({
            name: z.string(),
            link: z.string().url().optional(),
            image: z.string().url().optional(),
          })
        ),
      ])
      .optional(),
    repo: z.string().url().optional(),
  }),
});

export const collections = { posts };
