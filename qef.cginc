#define SVD_NUM_SWEEPS 5

// SVD
////////////////////////////////////////////////////////////////////////////////

const float Tiny_Number = 1.0e-20;

void givens_coeffs_sym(
    float a_pp, float a_pq, float a_qq, out float c, out float s)
{
    if (a_pq == 0.0)
    {
        c = 1.0;
        s = 0.0;
        return;
    }

    float tau = (a_qq - a_pp) / (2.0 * a_pq);
    float stt = sqrt(1.0 + tau * tau);
    float tan = 1.0 / ((tau >= 0.0) ? (tau + stt) : (tau - stt));
    c = rsqrt(1.0 + tan * tan);
    s = tan * c;
}

void svd_rotate_xy(inout float x, inout float y, float c, float s)
{
    float u = x; float v = y;
    x = c * u - s * v;
    y = s * u + c * v;
}

void svd_rotateq_xy(
    inout float x, inout float y, inout float a, float c, float s)
{
    float cc = c * c; float ss = s * s;
    float mx = 2.0 * c * s * a;
    float u = x; float v = y;
    x = cc * u - mx + ss * v;
    y = ss * u + mx + cc * v;
}

void svd_rotate(inout float3x3 vtav, inout float3x3 v, int a, int b)
{
    if (vtav[a][b] == 0.0)
    {
        return;
    }
    
    float c, s;
    givens_coeffs_sym(vtav[a][a], vtav[a][b], vtav[b][b], c, s);
    svd_rotateq_xy(vtav[a][a],vtav[b][b],vtav[a][b],c,s);
    svd_rotate_xy(vtav[0][3-b], vtav[1-a][2], c, s);
    vtav[a][b] = 0.0;
    
    svd_rotate_xy(v[0][a], v[0][b], c, s);
    svd_rotate_xy(v[1][a], v[1][b], c, s);
    svd_rotate_xy(v[2][a], v[2][b], c, s);
}

void svd_solve_sym(float3x3 a, out float3 sigma, inout float3x3 v)
{
    // assuming that A is symmetric: can optimize all operations for 
    // the upper right triagonal
    float3x3 vtav = a;

    // assuming V is identity: you can also pass a matrix the rotations
    // should be applied to
    // U is not computed
    for (int i = 0; i < SVD_NUM_SWEEPS; ++i)
    {
        svd_rotate(vtav, v, 0, 1);
        svd_rotate(vtav, v, 0, 2);
        svd_rotate(vtav, v, 1, 2);
    }

    sigma = float3(vtav[0][0],vtav[1][1],vtav[2][2]);    
}

float svd_invdet(float x, float tol)
{
    return (abs(x) < tol || abs(1.0 / x) < tol) ? 0.0 : (1.0 / x);
}

void svd_pseudoinverse(out float3x3 o, float3 sigma, float3x3 v)
{
    float d0 = svd_invdet(sigma[0], Tiny_Number);
    float d1 = svd_invdet(sigma[1], Tiny_Number);
    float d2 = svd_invdet(sigma[2], Tiny_Number);

    o = float3x3(
        v[0][0] * d0 * v[0][0] + v[0][1] * d1 * v[0][1] + v[0][2] * d2 * v[0][2],
        v[0][0] * d0 * v[1][0] + v[0][1] * d1 * v[1][1] + v[0][2] * d2 * v[1][2],
        v[0][0] * d0 * v[2][0] + v[0][1] * d1 * v[2][1] + v[0][2] * d2 * v[2][2],
        v[1][0] * d0 * v[0][0] + v[1][1] * d1 * v[0][1] + v[1][2] * d2 * v[0][2],
        v[1][0] * d0 * v[1][0] + v[1][1] * d1 * v[1][1] + v[1][2] * d2 * v[1][2],
        v[1][0] * d0 * v[2][0] + v[1][1] * d1 * v[2][1] + v[1][2] * d2 * v[2][2],
        v[2][0] * d0 * v[0][0] + v[2][1] * d1 * v[0][1] + v[2][2] * d2 * v[0][2],
        v[2][0] * d0 * v[1][0] + v[2][1] * d1 * v[1][1] + v[2][2] * d2 * v[1][2],
        v[2][0] * d0 * v[2][0] + v[2][1] * d1 * v[2][1] + v[2][2] * d2 * v[2][2]);
}

void svd_solve_ATA_ATb(float3x3 ATA, float3 ATb, out float3 x)
{
    float3x3 V = float3x3(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0);

    float3 sigma;
    
    svd_solve_sym(ATA, sigma, V);
    
    // A = UEV^T; U = A / (E*V^T)
    float3x3 Vinv;
    svd_pseudoinverse(Vinv, sigma, V);
    x = mul(Vinv, ATb);
}

float3 svd_vmul_sym(float3x3 a, float3 v)
{
    return float3(
        dot(a[0], v),
        (a[0][1] * v.x) + (a[1][1] * v.y) + (a[1][2] * v.z),
        (a[0][2] * v.x) + (a[1][2] * v.y) + (a[2][2] * v.z));
}

void svd_mul_ata_sym(out float3x3 o, float3x3 a)
{
    o[0][0] = a[0][0] * a[0][0] + a[1][0] * a[1][0] + a[2][0] * a[2][0];
    o[0][1] = a[0][0] * a[0][1] + a[1][0] * a[1][1] + a[2][0] * a[2][1];
    o[0][2] = a[0][0] * a[0][2] + a[1][0] * a[1][2] + a[2][0] * a[2][2];
    o[1][1] = a[0][1] * a[0][1] + a[1][1] * a[1][1] + a[2][1] * a[2][1];
    o[1][2] = a[0][1] * a[0][2] + a[1][1] * a[1][2] + a[2][1] * a[2][2];
    o[2][2] = a[0][2] * a[0][2] + a[1][2] * a[1][2] + a[2][2] * a[2][2];
}
    
void svd_solve_Ax_b(
    float3x3 a, float3 b, out float3x3 ATA, out float3 ATb, out float3 x)
{
    svd_mul_ata_sym(ATA, a);
    ATb = mul(b, a); // transpose(a) * b;
    svd_solve_ATA_ATb(ATA, ATb, x);
}

// QEF
////////////////////////////////////////////////////////////////////////////////

void qef_add(
    float3 n,
    float3 p,
    inout float3x3 ATA,
    inout float3 ATb,
    inout float4 pointaccum)
{
    ATA[0][0] += n.x * n.x;
    ATA[0][1] += n.x * n.y;
    ATA[0][2] += n.x * n.z;
    ATA[1][1] += n.y * n.y;
    ATA[1][2] += n.y * n.z;
    ATA[2][2] += n.z * n.z;

    float b = dot(p, n);
    ATb += n * b;
    pointaccum += float4(p,1.0);
}

float qef_calc_error(float3x3 A, float3 x, float3 b)
{
    float3 vtmp = b - svd_vmul_sym(A, x);
    return dot(vtmp,vtmp);
}

float qef_solve(float3x3 ATA, float3 ATb, float4 pointaccum, out float3 x)
{
    float3 masspoint = pointaccum.xyz / pointaccum.w;
    ATb -= svd_vmul_sym(ATA, masspoint);
    svd_solve_ATA_ATb(ATA, ATb, x);
    float result = qef_calc_error(ATA, x, ATb);
    
    x += masspoint;
        
    return result;
}
