import { Skeleton, Stack } from '@mui/material';

const GhostLoader: React.FC = () => {
    return (
        <Stack spacing={1} sx={{ width: '100%', my: 2 }}>
            <Skeleton
                variant="rectangular"
                height={200}
                animation="wave"
                sx={{ borderRadius: 2 }} // Add rounded corners
            />

            <Skeleton variant="text" animation="wave" sx={{ bgcolor: 'grey.600' }} />
            <Skeleton variant="text" animation="wave" width="75%" sx={{ bgcolor: 'grey.700' }} />

            {/* ... */}
        </Stack>
    );
};

export default GhostLoader;