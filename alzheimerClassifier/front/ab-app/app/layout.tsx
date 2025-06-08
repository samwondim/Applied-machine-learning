import type React from "react"
import "./globals.css"
import { Inter } from "next/font/google"
import { ThemeProvider } from "next-themes"

const inter = Inter({ subsets: ["latin"] })

export const metadata = {
  title: "Advanced Image Classification System",
  description: "AI-Powered Image Recognition with Real-time Analytics",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem suppressHydrationWarning>
          {children}
        </ThemeProvider>
      </body>
    </html>
  )
}
